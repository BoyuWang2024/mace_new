# MACE LLPR MAD、MATPES train 推理与 carnet 风格绘图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不重算曲率、不修改旧结果，为当前 MACE LLPR 增加显式共享曲率、通用邻居过滤和 carnet 风格四图，并完成 MATPES test 重绘及 MAD/MATPES train 远端正式计算。

**Architecture:** 保留现有 `build/calibrate/evaluate/validate`、canonical CSV 和结果身份体系。新增统一曲率源模块消除 calibration/inference 的重复加载逻辑；新绘图作为 MACE 内部第二种 style，复用现有 validation、只读快照、目录锁、manifest 和原子发布；MAD 过滤是独立的通用 extxyz 工具。

**Tech Stack:** Python、PyTorch、MACE、ASE、matscipy、NumPy、pandas、SciPy、Matplotlib、PyYAML、pytest、Slurm、Git。

## Global Constraints

- 只在 `Plots` 分支提交和推送，不融合 `main`。
- 使用 `conda activate mace_new`；命令从仓库根目录运行。
- 旧曲率、旧 Alpha、旧代码和 `legacy_matpes_r2scan` 结果只读，执行前后核对 SHA。
- 不重算曲率；旧 artifact 只读转换一次，正式代码只认识 canonical `base_curvature.pt`。
- 三个 variant 均用 fixed ridge `1.0e-12`；MATPES Alpha 用 `matpes_val.extxyz`，MAD Alpha 用过滤后的 `mad-val.xyz`。
- 能量为逐原子 `eV/atom`，力为逐笛卡尔分量 `eV/Å`。
- MAD 任一原子在 6.0 Å 内无邻居即排除整结构，val/test 都写完整 audit。
- `matpes_test` 仅重绘现有 validated canonical CSV，不进行推理。
- 禁止 import、调用、软链接或动态加载 `carnet_new` 模块；它只用于人工核对外观和固定数值契约。
- 大 CSV、PT、XYZ 和日志不进 Git；本地仅保留图、统计、manifest、validation 和 audit。
- 远端从 `origin/Plots` 建独立干净 worktree，不切换或清理现有工作区。
- 每项实现执行 red-green-refactor 并独立 commit；不暂存用户现有未跟踪文件。

## File Structure

- `llpr/config.py`：解析 `artifacts.curvature` 和 plot `style` 的上游计算配置。
- `llpr/curvature_source.py`：统一选择、哈希和验证内部/外部曲率。
- `llpr/calibration.py`、`llpr/inference.py`、`llpr/validation.py`：消费并传播曲率来源身份。
- `llpr/dataset_filter.py` 与 `scripts/filter_neighborless_extxyz.py`：通用 extxyz 过滤与 audit。
- `llpr/density_plotting.py`：MACE 内部独立实现密度分析和四图渲染。
- `llpr/plotting.py`、`llpr/cli.py`：复用现有事务式绘图流程并按 style 分派。
- `configs/`：n20、MATPES test、MAD、MATPES train 正式配置。
- `scripts/run_remote_stage.sh`、`scripts/submit_remote_stage.slurm`：通用远端阶段执行。
- `tests/`：每个新接口的单元、集成和 n20 全链路覆盖。

---

### Task 1: 严格解析显式共享曲率配置

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/config.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py`

**Interfaces:**
- Consumes: existing `PathIdentity` and `_path_identity()`.
- Produces: `ArtifactsConfig(curvature: PathIdentity | None)` and `LLPRConfig.artifacts`.

- [ ] **Step 1: Write failing tests**

```python
def test_load_config_resolves_optional_curvature_artifact(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace(
        "curvature:\n",
        f"artifacts:\n  curvature:\n    path: shared/base_curvature.pt\n"
        f"    expected_sha256: {'5' * 64}\ncurvature:\n",
    ), encoding="utf-8")
    config = load_config(config_path)
    assert config.artifacts.curvature.path == (tmp_path / "shared/base_curvature.pt").resolve()
    assert config.artifacts.curvature.expected_sha256 == "5" * 64
```

增加无 `artifacts` 保持旧行为、缺 SHA、非法 SHA、空路径和多余字段的用例。

- [ ] **Step 2: Verify RED**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py
```

Expected: new assertions fail because `LLPRConfig.artifacts` does not exist.

- [ ] **Step 3: Implement minimal schema**

```python
@dataclass(frozen=True)
class ArtifactsConfig:
    curvature: PathIdentity | None = None

@dataclass(frozen=True)
class LLPRConfig:
    artifacts: ArtifactsConfig = ArtifactsConfig()
```

只允许 `artifacts.curvature.{path,expected_sha256}`；section 可缺失，显式 artifact 的 SHA 不可为空。

- [ ] **Step 4: Verify and commit**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py Uncertainty_Quantification/LLPR/tests/test_curvature_ridge.py

git add Uncertainty_Quantification/LLPR/llpr/config.py Uncertainty_Quantification/LLPR/tests/test_artifacts_config.py
git commit -m "feat(llpr): parse shared curvature artifacts"
```

### Task 2: 统一曲率加载、SHA 和 provenance

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/curvature_source.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_curvature_source.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/validation.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_calibration.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_inference_validation.py`

**Interfaces:**
- Produces: `LoadedCurvature`, `curvature_source_path()` and `load_curvature_source()`.

- [ ] **Step 1: Write failing source tests**

```python
def test_external_sha_is_checked_before_deserialization(config, monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("deserialized"))
    with pytest.raises(ValueError, match="curvature artifact SHA256 mismatch"):
        load_curvature_source(config, checkpoint, layout)

def test_internal_source_remains_default(config):
    assert curvature_source_path(config, checkpoint_sha) == run_root(config, checkpoint_sha) / "curvature/base_curvature.pt"
```

覆盖普通非空文件、checkpoint/readout、CPU float64、shape、有限性、对称性和 `hef == he + hf`。

- [ ] **Step 2: Verify RED**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_curvature_source.py
```

- [ ] **Step 3: Implement shared loader**

```python
@dataclass(frozen=True)
class LoadedCurvature:
    path: Path
    sha256: str
    identity: Mapping[str, Any]
    variants: Mapping[str, Tensor]
```

先 `lstat` 和 SHA，后 `torch.load`。calibration/inference 删除各自 `_load_curvature()`，共同调用新模块；`run_build()` 永远只写本次 run root。

- [ ] **Step 4: Propagate and validate provenance**

```python
curvature_artifact = {"path": str(loaded.path.resolve()), "sha256": loaded.sha256}
```

把该对象写入 calibration/evaluation progress identity；validation 严格检查 `{path, sha256}` 和跨层一致性，但回传后不要求 PT 仍在本地。

- [ ] **Step 5: Verify full related suite and commit**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_curvature_source.py Uncertainty_Quantification/LLPR/tests/test_calibration.py Uncertainty_Quantification/LLPR/tests/test_inference_validation.py

git add Uncertainty_Quantification/LLPR/llpr/curvature_source.py \
  Uncertainty_Quantification/LLPR/llpr/calibration.py \
  Uncertainty_Quantification/LLPR/llpr/inference.py \
  Uncertainty_Quantification/LLPR/llpr/validation.py \
  Uncertainty_Quantification/LLPR/tests/test_curvature_source.py \
  Uncertainty_Quantification/LLPR/tests/test_calibration.py \
  Uncertainty_Quantification/LLPR/tests/test_inference_validation.py
git commit -m "feat(llpr): reuse validated curvature sources"
```

### Task 3: 通用 extxyz 邻居过滤与 audit

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/dataset_filter.py`
- Create: `Uncertainty_Quantification/LLPR/scripts/filter_neighborless_extxyz.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_dataset_filter.py`

**Interfaces:**
- Produces: `missing_neighbor_indices(atoms, cutoff)` and `filter_neighborless_extxyz(...)`.

- [ ] **Step 1: Write failing tests**

```python
def test_missing_neighbor_excludes_whole_structure() -> None:
    atoms = Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [20, 0, 0]])
    assert missing_neighbor_indices(atoms, 6.0) == (2,)

def test_filter_writes_hash_complete_audit(tmp_path: Path) -> None:
    report = filter_neighborless_extxyz(source, output, audit, cutoff=6.0)
    assert report["source_sha256"] == sha256_file(source)
    assert report["output_sha256"] == sha256_file(output)
    assert report["total_structures"] == report["retained_structures"] + report["excluded_structures"]
```

覆盖单原子、周期邻居、非周期孤立原子、写入失败清理和确定性重跑。

- [ ] **Step 2: Verify RED, implement, verify GREEN**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_dataset_filter.py
```

实现使用 `matscipy.neighbours.neighbour_list("i", ...)`，流式 ASE 读写；任一缺失原子排除整结构。audit 记录 source index、structure id、原子数、缺失索引、PBC、元素和原因，临时文件验证后原子替换。

- [ ] **Step 3: Add thin CLI and commit**

```bash
python Uncertainty_Quantification/LLPR/scripts/filter_neighborless_extxyz.py input.extxyz output.extxyz audit.json --cutoff 6.0

git add Uncertainty_Quantification/LLPR/llpr/dataset_filter.py Uncertainty_Quantification/LLPR/scripts/filter_neighborless_extxyz.py Uncertainty_Quantification/LLPR/tests/test_dataset_filter.py
git commit -m "feat(llpr): audit neighborless extxyz structures"
```

### Task 4: MACE 内部独立实现 `carnet_density`

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/density_plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/cli.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`

**Interfaces:**
- Consumes: current validated snapshot and `_PanelData`.
- Produces: four fixed figures, four-row statistics and style-aware manifest.

- [ ] **Step 1: Write failing style and independence tests**

```python
def test_plot_style_defaults_to_existing_suite(tmp_path: Path) -> None:
    assert _load_plot_config(write_plot_config(tmp_path)).style == "diagnostic_suite"

def test_mace_plotting_has_no_carnet_runtime_dependency() -> None:
    sources = [Path(plotting.__file__), Path(density_plotting.__file__)]
    assert all("carnet_new" not in path.read_text(encoding="utf-8") for path in sources)
```

`carnet_density` 必须精确选择 `he/energy`、`hf/forces`、`hef/energy`、`hef/forces`；未知 style 和多余字段报错。

- [ ] **Step 2: Write failing numerical contract tests**

```python
def test_pair_limits_are_dataset_local() -> None:
    limits = carnet_shared_limits(panels, margin=0.05)
    assert limits[("he", "energy")] == limits[("hef", "energy")]
    assert limits[("hf", "forces")] == limits[("hef", "forces")]

def test_sampling_is_deterministic() -> None:
    assert np.array_equal(deterministic_sample_indices(50000, 20000, 20260714), deterministic_sample_indices(50000, 20000, 20260714))
```

覆盖零 residual、非正 std、Pearson/Spearman、160 网格、sigma 1.2、五个 contour mass 和排除计数。

- [ ] **Step 3: Implement locally without calling carnet code**

```python
CARNET_DENSITY_CONFIG = {
    "grid_size": 160, "gaussian_sigma": 1.2,
    "contour_masses": (0.5, 0.7, 0.85, 0.95, 0.99),
    "scatter_max_points": 20000, "random_seed": 20260714,
    "log_margin": 0.05, "figure_size": (7.0, 7.0),
    "color": "#f28e2b", "scatter_size": 12.0,
    "scatter_alpha": 0.04, "dpi": 300,
}
```

仅依据已核对契约在 `mace_new` 内实现算法；不把 carnet 路径加入 `sys.path`，不 import、subprocess 调用或链接 carnet 文件。

- [ ] **Step 4: Reuse existing transactional pipeline**

```python
def run_plot(publication_root: Path, *, output_dir: Path, selected=DEFAULT_SELECTED, style="diagnostic_suite", dpi=None) -> Path:
    # both styles use the same validation, snapshot, lock and atomic promotion
```

旧 style 保持六图/180 dpi；新 style 严格生成四 PNG、四 PDF、statistics 和 manifest。He/Hef energy 共享范围，Hf/Hef force 共享范围，不跨数据集。

- [ ] **Step 5: Verify and commit**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_plotting.py

git add Uncertainty_Quantification/LLPR/llpr/density_plotting.py Uncertainty_Quantification/LLPR/llpr/plotting.py Uncertainty_Quantification/LLPR/llpr/cli.py Uncertainty_Quantification/LLPR/tests/test_plotting.py
git commit -m "feat(llpr): add independent density publication plots"
```

### Task 5: n20 全链路和正式运维接口

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py`
- Create: `Uncertainty_Quantification/LLPR/configs/cpu_n20_shared_curvature.yaml`
- Create: three `plot_carnet_*.yaml` configs.
- Create: `Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh`
- Create: `Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm`
- Modify: `Uncertainty_Quantification/LLPR/README.md`, `.gitignore`.

- [ ] **Step 1: Add a failing build-once/consume-once n20 test**

先用现有 n20 config build，再用第二 experiment 显式指向其 artifact，执行 calibrate/evaluate/validate/plot；断言 consumer 不生成 curvature 文件、validation valid、恰有四 PNG/PDF。

- [ ] **Step 2: Implement configs and generic stage scripts**

```bash
bash Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh calibrate Uncertainty_Quantification/LLPR/configs/cpu_n20_shared_curvature.yaml
sbatch Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm evaluate Uncertainty_Quantification/LLPR/configs/cpu_n20_shared_curvature.yaml
```

脚本只允许 `calibrate/evaluate/validate/plot`，使用 `conda run -n mace_new`、`set -euo pipefail`，绝不隐式 build。

- [ ] **Step 3: Run n20 and full suite**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py -s
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests
bash -n Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh
bash -n Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore \
  Uncertainty_Quantification/LLPR/README.md \
  Uncertainty_Quantification/LLPR/configs/cpu_n20_shared_curvature.yaml \
  Uncertainty_Quantification/LLPR/configs/plot_carnet_matpes_test.yaml \
  Uncertainty_Quantification/LLPR/configs/plot_carnet_matpes_train.yaml \
  Uncertainty_Quantification/LLPR/configs/plot_carnet_mad_test.yaml \
  Uncertainty_Quantification/LLPR/scripts/run_remote_stage.sh \
  Uncertainty_Quantification/LLPR/scripts/submit_remote_stage.slurm \
  Uncertainty_Quantification/LLPR/tests/test_n20_full_chain.py \
  Uncertainty_Quantification/LLPR/tests/test_remote_scripts.py
git commit -m "ops(llpr): add shared-curvature publication workflow"
```

### Task 6: 使用现有 MATPES test 结果重绘

- [ ] **Step 1: Revalidate without inference**

对 `/home/lilong/code/UQ/mace_new/Uncertainty_Quantification/LLPR/outputs/legacy_matpes_r2scan/8f147ecffa1d/evaluation/deterministic` 调用 `validate_publication_root()`，不改 canonical CSV。

- [ ] **Step 2: Run the new plot config**

```bash
conda run -n mace_new python -m Uncertainty_Quantification.LLPR.llpr plot --config Uncertainty_Quantification/LLPR/configs/plot_carnet_matpes_test.yaml
```

- [ ] **Step 3: Verify exact ten-file contract**

四 PNG、四 PDF、`plotting_statistics.csv`、`plotting_manifest.json`；manifest 输入 SHA 与现有 canonical 结果一致。

### Task 7: 远端转换、过滤、正式配置和作业

- [ ] **Step 1: Push and create clean remote worktree**

```bash
git push origin Plots
ssh -p 55801 bywang@121.48.164.204 'git -C /home/bywang/code/UQ/mace_new fetch origin Plots && git -C /home/bywang/code/UQ/mace_new worktree add /home/bywang/code/UQ/mace_new-plots origin/Plots'
```

已存在 worktree 时只检查分支/状态，不 reset 或删除。

- [ ] **Step 2: Record old hashes and convert read-only**

记录旧曲率与全部旧 Alpha SHA。远端临时程序验证 `HE/HF/Hef/Hef_reg`、shape 2192、CPU float64、对称、`Hef=HE+HF`、`Hef_reg=Hef+1e-12I`，只写 canonical `HE/HF/Hef` 和 conversion audit；复验后删除临时程序。

- [ ] **Step 3: Upload and filter exact MAD files**

```bash
scp -P 55801 /home/lilong/code/UQ/upet_new/data/dataset/mad-val.xyz /home/lilong/code/UQ/upet_new/data/dataset/mad-test.xyz bywang@121.48.164.204:/home/bywang/code/UQ/mace_new-plots/data/dataset/
```

核对原始 SHA，运行 6.0 Å 过滤器并记录过滤后 SHA/audit。

- [ ] **Step 4: Commit formal configs with measured hashes**

创建 `gpu_matpes_train_shared_curvature.yaml` 和 `gpu_mad_shared_curvature.yaml`，填写实际 canonical curvature SHA 和实际过滤后 MAD SHA；两者 ridge 为 `1.0e-12`。提交、推送，远端干净 worktree fast-forward。

- [ ] **Step 5: Run capped smoke then submit formal dependencies**

用独立 smoke experiment 设置 `max_structures: 2`、`max_force_components_per_structure: 3`，完成 calibrate/evaluate/validate/plot。通过后按 `calibrate -> evaluate -> validate -> plot` 的 `afterok` 链提交两套正式作业并监控到终态。

- [ ] **Step 6: Verify production and preservation**

核对两套 validation、三 variant 行数、Alpha/ridge、输入/输出 SHA、每数据集十个绘图文件；重新计算全部旧文件 SHA，必须与执行前一致。

### Task 8: 回传白名单、视觉 QA 和最终提交

- [ ] **Step 1: Transfer whitelist only**

回传 MAD/MATPES train 的四 PNG、四 PDF、statistics、plot manifest、validation、filter/conversion audit 和小型 job summary；不回传 PT、大 CSV、filtered XYZ 或日志全集。

- [ ] **Step 2: Hash and visual verification**

按 manifest 重算 SHA；检查全部 12 张 PNG 的字体、轮廓、灰区、`y=x`、注释、裁切和空白；检查 PDF header/trailer。

- [ ] **Step 3: Fresh final verification**

```bash
conda run -n mace_new pytest -q Uncertainty_Quantification/LLPR/tests
git diff --check
git status --short
```

核对三个 plot manifest 的 style、固定参数、四组选择、validation SHA 和 24 个图像 SHA。

- [ ] **Step 4: Commit and push only tracked deliverables**

```bash
git add .gitignore Uncertainty_Quantification/LLPR docs/superpowers
git diff --cached --check
git commit -m "docs(llpr): finalize remote publication workflow"
git push origin Plots
```

生成图保持 ignored，不使用 `git add -f`；用户未跟踪文件保持原样。

## Self-Review

- Spec coverage: Tasks 1–2 覆盖共享曲率和身份；Task 3 覆盖 MAD 整结构过滤；Task 4 覆盖独立四图；Task 5 覆盖小数据全链路；Tasks 6–8 覆盖重绘、远端计算、回传和 QA。
- Placeholder scan: 正式 SHA 在远端测量后才写入 production configs，不提交假 SHA；测试中的重复字符仅是 fixture。
- Type consistency: canonical target 始终为 `energy`/`forces`，仅图文件 stem 使用 singular `force`。
- Independence: MACE 代码不直接调用 carnet 模块；测试显式扫描该依赖。
- Reuse: canonical CSV、validation、snapshot、locking、atomic promotion、manifest、run root 和四个计算阶段保持唯一权威实现。
