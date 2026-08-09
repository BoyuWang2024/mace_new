# FGE Remote Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在新仓库中实现可复现的 FGE 论文标准绘图入口，在远端为四组正式结果生成 43 张逻辑图，并仅把 86 个 PNG/PDF 图片文件回传到本地。

**Architecture:** 绘图代码直接读取当前 canonical FGE artifact，不读取 extxyz、不加载模型、不训练、不预测。数据读取、确定性密度分析、单实验渲染、跨实验比较和原子发布分别放在聚焦模块中；远端 figures 目录构建完成并通过审计后一次性发布。

**Tech Stack:** Python 3.11、PyTorch、NumPy、SciPy、Matplotlib、pytest、现有 `mace` conda 环境、SSH/rsync。

## Global Constraints

- 所有测试和实际绘图只在远端执行，本地不运行 Python 测试或绘图。
- 使用远端现有 `mace` 环境；不克隆或复制 conda 环境。
- 不重新训练、不重新预测、不加载模型、不读取 extxyz。
- 只读取四组 `PASS` 正式结果的 canonical artifact。
- 只生成 Energy/Force；不得生成 Stress 图。
- 同时绘制 `equal_weight` 与 `validation_weighted`。
- PNG 固定 300 DPI，PDF 保留矢量文字/坐标轴并栅格化大规模散点。
- uncertainty–residual 复用参考图视觉规范：`#f28e2b`、log-log、密度等高线、灰色安全区、`y=x`、相关系数标注。
- 随机种子固定为 `20260714`，散点最多确定性抽样 `20_000` 点，密度使用全部有效点。
- 图片写入 `Uncertainty_Quantification/FGE/outputs/figures/`，不得修改正式结果树。
- 最终恰好生成 43 张逻辑图、86 个图片文件；审计 JSON 只保留远端。
- 本地仅拉取 `*.png` 和 `*.pdf`，不得拉取 JSON、CSV、张量、模型或日志。

---

### Task 1: Canonical Plot Data Boundary

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/plot_data.py`
- Test: `Uncertainty_Quantification/FGE/tests/test_plot_data.py`

**Interfaces:**
- Consumes: canonical result root containing `prediction/test_raw.pt`, both `evaluation/<branch>/` trees, `validation.json`, and `result_manifest.json`.
- Produces: `load_plot_run(root: Path) -> PlotRun`, with `PlotRun.branches: dict[str, BranchPlotData]`; `BranchPlotData` contains aligned CPU float64 Energy/Force reference, prediction, residual, STD, metrics, correlations, and risk rows.

- [ ] **Step 1: Write failing tests for PASS status, branch alignment, and per-atom Energy**

```python
def test_load_plot_run_builds_aligned_energy_and_force_data(result_fixture):
    run = load_plot_run(result_fixture)
    branch = run.branches["equal_weight"]
    assert torch.equal(branch.energy_reference, torch.tensor([1.0, 2.0]))
    assert torch.equal(branch.energy_prediction, torch.tensor([1.5, 2.5]))
    assert torch.equal(branch.energy_residual, torch.tensor([0.5, 0.5]))
    assert branch.force_reference.shape == branch.force_prediction.shape

def test_load_plot_run_rejects_non_pass_result(result_fixture):
    (result_fixture / "validation.json").write_text('{"status":"FAIL"}')
    with pytest.raises(HardFailure, match="PASS"):
        load_plot_run(result_fixture)
```

- [ ] **Step 2: Run the new test remotely and verify RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_data.py -q`

Expected: collection fails because `fge.plot_data` does not exist.

- [ ] **Step 3: Implement strict canonical loading**

```python
@dataclass(frozen=True)
class BranchPlotData:
    energy_reference: Tensor
    energy_prediction: Tensor
    energy_residual: Tensor
    energy_uncertainty: Tensor
    force_reference: Tensor
    force_prediction: Tensor
    force_residual: Tensor
    force_uncertainty: Tensor
    metrics: Mapping[str, Any]
    correlations: tuple[Mapping[str, Any], ...]
    risk_coverage: tuple[Mapping[str, Any], ...]

def load_plot_run(root: Path) -> PlotRun:
    # Require PASS markers, load weights_only tensors, require CPU float64,
    # divide total Energy by n_atoms, flatten Force components, and align shapes.
```

The loader must reject Stress requests and must never import training, prediction inference, data loaders, ASE, or MACE model loading.

- [ ] **Step 4: Run plot-data and existing validation tests remotely**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_data.py Uncertainty_Quantification/FGE/tests/test_validation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add Uncertainty_Quantification/FGE/fge/plot_data.py Uncertainty_Quantification/FGE/tests/test_plot_data.py
git commit -m "feat(fge): load canonical plotting data"
```

### Task 2: Deterministic Density and Rendering Primitives

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/plot_density.py`
- Create: `Uncertainty_Quantification/FGE/fge/plot_style.py`
- Test: `Uncertainty_Quantification/FGE/tests/test_plot_density.py`

**Interfaces:**
- Consumes: aligned 1-D CPU float64 tensors.
- Produces: `analyze_log_panel(uncertainty: Tensor, residual: Tensor, config: PlotConfig) -> PanelAnalysis`, `deterministic_indices(count: int, maximum: int, seed: int) -> Tensor`, and immutable `PlotConfig`.

- [ ] **Step 1: Write failing tests for filtering and deterministic sampling**

```python
def test_log_analysis_records_nonpositive_and_nonfinite_values():
    result = analyze_log_panel(
        torch.tensor([1.0, 0.0, float("inf"), 2.0]),
        torch.tensor([1.0, 3.0, 4.0, 2.0]),
        PlotConfig(),
    )
    assert result.valid_count == 2
    assert result.excluded == {"zero": 1, "negative": 0, "nan": 0, "inf": 1}

def test_sampling_is_stable_and_bounded():
    first = deterministic_indices(100_000, 20_000, 20260714)
    second = deterministic_indices(100_000, 20_000, 20260714)
    assert first.numel() == 20_000
    assert torch.equal(first, second)
```

- [ ] **Step 2: Run remotely and verify RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_density.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement fixed plotting configuration and density analysis**

```python
@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 300
    scatter_max_points: int = 20_000
    scatter_size: float = 2.0
    scatter_alpha: float = 0.035
    random_seed: int = 20260714
    grid_size: int = 160
    gaussian_sigma: float = 1.2
    contour_masses: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)
```

Use `numpy.histogram2d` on all valid log10 points and `scipy.ndimage.gaussian_filter`; derive strictly increasing contour levels and compute Pearson(log10)/Spearman on all valid points.

- [ ] **Step 4: Run density tests remotely**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_density.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add Uncertainty_Quantification/FGE/fge/plot_density.py Uncertainty_Quantification/FGE/fge/plot_style.py Uncertainty_Quantification/FGE/tests/test_plot_density.py
git commit -m "feat(fge): add deterministic plot analysis"
```

### Task 3: Single-Experiment Figure Renderer

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/plot_single.py`
- Test: `Uncertainty_Quantification/FGE/tests/test_plot_single.py`

**Interfaces:**
- Consumes: `PlotRun`, branch name, output directory, `PlotConfig`.
- Produces: `render_single_run(run: PlotRun, output_root: Path, config: PlotConfig) -> tuple[FigureRecord, ...]`, exactly five logical figures per branch and ten PNG/PDF files per branch.

- [ ] **Step 1: Write failing rendering contract test**

```python
def test_render_single_run_writes_five_png_pdf_pairs_per_branch(plot_run, tmp_path):
    records = render_single_run(plot_run, tmp_path, PlotConfig(dpi=72))
    images = sorted(path.suffix for path in tmp_path.rglob("*.*") if path.suffix in {".png", ".pdf"})
    assert len(records) == 10
    assert images.count(".png") == 10
    assert images.count(".pdf") == 10
```

- [ ] **Step 2: Run remotely and verify RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_single.py -q`

Expected: missing renderer failure.

- [ ] **Step 3: Implement Agg-only rendering**

Implement:

- square Energy and Force parity density panels with equality line;
- reference-style Energy and Force uncertainty–residual panels;
- two-panel Energy/Force risk-coverage figure;
- PNG/PDF paired saving through sibling temporary files;
- English labels and consistent branch subtitle;
- rasterized scatter collections for PDF.

The uncertainty panel must use `ORANGE = "#f28e2b"`, gray safe region, black dashed equality line, shared log limits, and correlation annotation.

- [ ] **Step 4: Run renderer tests and inspect test images remotely**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_single.py -q`

Expected: PASS and every image has nonzero size.

- [ ] **Step 5: Commit Task 3**

```bash
git add Uncertainty_Quantification/FGE/fge/plot_single.py Uncertainty_Quantification/FGE/tests/test_plot_single.py
git commit -m "feat(fge): render publication single-run figures"
```

### Task 4: Cross-Experiment Comparison and Atomic Workflow

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/plot_compare.py`
- Create: `Uncertainty_Quantification/FGE/fge/plot_workflow.py`
- Create: `Uncertainty_Quantification/FGE/scripts/plot.py`
- Test: `Uncertainty_Quantification/FGE/tests/test_plot_workflow.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_scripts.py`

**Interfaces:**
- Consumes: ordered mapping of four experiment labels to canonical roots and output root.
- Produces: `render_all_figures(result_roots: Mapping[str, Path], output_root: Path) -> Path`; publishes exactly 43 PNG and 43 PDF files plus remote-only `plot_audit.json`.

- [ ] **Step 1: Write failing workflow tests**

```python
def test_workflow_publishes_exact_figure_contract(four_plot_runs, tmp_path):
    final = render_all_figures(four_plot_runs, tmp_path / "figures", config=PlotConfig(dpi=72))
    assert len(list(final.rglob("*.png"))) == 43
    assert len(list(final.rglob("*.pdf"))) == 43
    assert (final / "plot_audit.json").is_file()

def test_workflow_does_not_modify_formal_results(four_result_roots, tmp_path):
    before = {root: sha256_file(root / "result_manifest.json") for root in four_result_roots}
    render_all_figures(four_result_roots, tmp_path / "figures", config=PlotConfig(dpi=72))
    after = {root: sha256_file(root / "result_manifest.json") for root in four_result_roots}
    assert before == after
```

- [ ] **Step 2: Run remotely and verify RED**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_workflow.py Uncertainty_Quantification/FGE/tests/test_scripts.py -q`

Expected: missing workflow/script failure.

- [ ] **Step 3: Implement three comparison figures and staging publication**

Comparison figures:

1. 2-panel Energy/Force RMSE grouped comparison;
2. 2×2 Energy/Force × Pearson/Spearman comparison;
3. 2-panel Energy/Force risk-coverage overlays.

Write all output to `.<name>.staging-<uuid>`, validate 43 PNG + 43 PDF and audit contents, then use one `os.replace` to publish. Refuse to overwrite an existing completed figures directory.

- [ ] **Step 4: Implement explicit script arguments**

```python
parser.add_argument("--result", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
parser.add_argument("--output-root", type=Path, required=True)
```

No central package CLI is added.

- [ ] **Step 5: Run workflow and script tests remotely**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests/test_plot_workflow.py Uncertainty_Quantification/FGE/tests/test_scripts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add Uncertainty_Quantification/FGE/fge/plot_compare.py Uncertainty_Quantification/FGE/fge/plot_workflow.py Uncertainty_Quantification/FGE/scripts/plot.py Uncertainty_Quantification/FGE/tests/test_plot_workflow.py Uncertainty_Quantification/FGE/tests/test_scripts.py
git commit -m "feat(fge): orchestrate complete plotting suite"
```

### Task 5: Documentation, Remote Production, and Image-Only Transfer

**Files:**
- Modify: `Uncertainty_Quantification/FGE/README.md`
- Verify: `Uncertainty_Quantification/FGE/outputs/.gitignore`

**Interfaces:**
- Consumes: completed plotting script and four remote formal roots.
- Produces: remote `outputs/figures/`, local ignored mirror containing only 43 PNG and 43 PDF files.

- [ ] **Step 1: Document plotting invocation and exclusions**

Document the four explicit `--result LABEL PATH` arguments, output root, Energy/Force-only constraint, and that plotting never trains or predicts.

- [ ] **Step 2: Run the full remote test suite**

Run: `python -m pytest -p no:cacheprovider Uncertainty_Quantification/FGE/tests Uncertainty_Quantification/FGE/internal_migration/tests -q`

Expected: all tests PASS; only known upstream warnings are allowed.

- [ ] **Step 3: Record formal manifest hashes before production plotting**

Run remotely: hash each `validation.json` and `result_manifest.json` into a temporary shell report outside all formal roots.

- [ ] **Step 4: Generate all figures remotely in the existing mace environment**

Run `python -m Uncertainty_Quantification.FGE.scripts.plot` with four explicit result pairs and `--output-root Uncertainty_Quantification/FGE/outputs/figures`.

Expected: atomic publication containing 43 PNG, 43 PDF, and one audit JSON.

- [ ] **Step 5: Verify remote images and formal-root immutability**

Check:

- exactly 43 `*.png` and 43 `*.pdf`;
- PNG signature `89504e470d0a1a0a`;
- PDF signature `%PDF-`;
- all files nonempty;
- PNG DPI metadata approximately 300;
- before/after formal manifest hashes identical;
- no `.staging-*` directory remains.

- [ ] **Step 6: Pull only pictures to local ignored output**

Use two rsync include filters: one for `*.png`, one for `*.pdf`, with all other files excluded. Destination:

`Uncertainty_Quantification/FGE/outputs/figures/`

- [ ] **Step 7: Verify local transfer without local plotting**

Check only filesystem properties and hashes: 43 PNG, 43 PDF, no JSON/CSV/PT/model/log files, and sampled local SHA-256 values equal remote values.

- [ ] **Step 8: Commit documentation**

```bash
git add Uncertainty_Quantification/FGE/README.md
git commit -m "docs(fge): document remote plotting workflow"
```

- [ ] **Step 9: Final repository check**

Run `git status --short`, confirm only pre-existing unrelated untracked paths remain and `outputs/figures/` is ignored.
