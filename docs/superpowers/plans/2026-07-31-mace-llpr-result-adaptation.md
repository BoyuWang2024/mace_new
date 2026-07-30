# MACE LLPR 旧结果一次性适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改旧目录、不重新运行模型且不保留迁移代码的前提下，把 7 月六个旧 details PT 适配为正式 LLPR 的统一发布结果包。

**Architecture:** 使用仅存在于执行期间的一次性转换程序，从只读旧 PT 写入新仓库 staging 目录。转换完成后调用正式 `validation.py` 和 `plotting.py`，通过全部校验后原子发布并删除转换程序；审计证据保存在发布目录之外。

**Tech Stack:** Python 3.10、PyTorch、pandas、正式 MACE LLPR validation/plotting API。

## Global Constraints

- 前置计划 `docs/superpowers/plans/2026-07-31-mace-llpr-implementation.md` 必须全部完成。
- 旧根目录 `/home/lilong/code/UQ/mace/UQ_LLPR/matpes_r2` 始终只读。
- 唯一输入是顶层 `results/{He,Hf,Hef}` 六个 details PT。
- 旧绘图、旧 summary、旧 matrix 和旧 alpha 文件均不迁移。
- 不运行 checkpoint，不读取模型，不重新构建曲率，不重新校准 alpha。
- He/Hef Energy 直接使用显式逐原子字段；Hf Energy 执行可逆的 `N`/`N²` 单位变换。
- Force component 不重排、不聚合、不抽样。
- 一次性转换代码不提交，最终工作区不得保留该文件。
- 审计目录位于正式 `evaluation/deterministic/` 之外。

---

## Task 1: 固定源文件身份和 staging 安全边界

**Files:**
- Temporary create: `Uncertainty_Quantification/LLPR/.local_adaptation/adapt_results.py`
- Create then keep untracked: `Uncertainty_Quantification/LLPR/outputs/_audit/<run_id>/source_manifest.json`
- No source modifications.

**Interfaces:**
- Consumes: 六个旧 PT。
- Produces: 源 SHA256 manifest、staging 路径和只读检查结果。

- [ ] **Step 1: 记录迁移前 Git 与源 SHA256**

Run:

```bash
git -C /home/lilong/code/UQ/mace status --short
sha256sum \
  /home/lilong/code/UQ/mace/UQ_LLPR/matpes_r2/results/He/*details.pt \
  /home/lilong/code/UQ/mace/UQ_LLPR/matpes_r2/results/Hf/*details.pt \
  /home/lilong/code/UQ/mace/UQ_LLPR/matpes_r2/results/Hef/*details.pt
```

把六个绝对路径、大小和 SHA256 写入审计 manifest。记录旧仓库原有未跟踪文件，
后续不得把它们误判为本任务修改。

- [ ] **Step 2: 验证唯一输入集合**

一次性程序固定：

```python
SOURCE_FILES = {
    (variant, target): SOURCE_ROOT / variant_name / filename
    for variant, variant_name in {"he": "He", "hf": "Hf", "hef": "Hef"}.items()
    for target, filename in {
        "energy": "llpr_energy_prediction_details.pt",
        "forces": "llpr_force_prediction_details.pt",
    }.items()
}
```

要求集合恰有六个文件，全部存在，且 SHA256 与 source manifest 一致。

- [ ] **Step 3: 创建目标 staging**

目标：

```text
Uncertainty_Quantification/LLPR/outputs/
└── legacy_matpes_r2scan/<checkpoint_sha12>/
    └── evaluation/.deterministic.staging-<run_id>/
```

程序启动时拒绝覆盖已存在 staging 或正式 `deterministic/`。路径必须
`resolve()` 后仍位于新仓库 `Uncertainty_Quantification/LLPR/outputs` 内。

- [ ] **Step 4: 运行 dry inspection**

只加载六个 PT，断言：

```python
assert len(payload) == 19_374
assert isinstance(payload[0], dict)
```

打印字段集合、结构数、Force 总分量数和源 SHA256，不写 CSV。

- [ ] **Step 5: 保存审计 checkpoint**

审计 manifest 写入：

```json
{
  "status": "source_verified",
  "source_files": [],
  "expected_structures_per_variant": 19374,
  "expected_force_components_per_variant": 447963
}
```

## Task 2: TDD 实现逐值字段映射

**Files:**
- Temporary modify: `Uncertainty_Quantification/LLPR/.local_adaptation/adapt_results.py`
- Temporary create: `Uncertainty_Quantification/LLPR/.local_adaptation/test_adapt_results.py`

**Interfaces:**
- Consumes: 单条旧 Energy/Force dict。
- Produces: `map_energy_row`, `map_force_rows`, `recover_alpha`。

- [ ] **Step 1: 写 He/Hef 和 Hf Energy 映射测试**

```python
def test_map_explicit_per_atom_energy():
    row = map_energy_row("he", explicit_energy_record(), canonical_id="319121")
    assert row["reference"] == explicit_energy_record()["energy_true_per_atom"]
    assert row["q"] == explicit_energy_record()["score_per_atom"]


def test_map_hf_total_energy_by_n_and_n_squared():
    old = hf_energy_record(natoms=19)
    row = map_energy_row("hf", old, canonical_id="319121")
    assert row["reference"] == old["ref_energy"] / 19
    assert row["q"] == old["llpr_score_energy"] / 19**2
    assert row["variance"] == old["llpr_var_energy"] / 19**2
    assert row["std"] == old["llpr_std_energy"] / 19
```

- [ ] **Step 2: 写 Force 索引和 alpha 恢复测试**

```python
def test_force_flat_index_maps_to_atom_and_direction():
    rows = map_force_rows("he", force_record(indices=[0, 4, 8]), "319121")
    assert [(r["atom_index"], r["direction"]) for r in rows] == [
        (0, 0), (1, 1), (2, 2)
    ]


def test_recover_alpha_requires_constant_ratio():
    assert recover_alpha(q=[1.0, 4.0], std=[2.0, 4.0]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="not constant"):
        recover_alpha(q=[1.0, 4.0], std=[2.0, 5.0])
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```bash
python -m pytest \
  Uncertainty_Quantification/LLPR/.local_adaptation/test_adapt_results.py -v
```

Expected: FAIL，映射函数尚不存在。

- [ ] **Step 4: 实现映射并运行测试**

Energy 返回字段顺序必须等于正式 `ENERGY_FIELDS`。Force 返回字段顺序必须等于
正式 `FORCE_FIELDS`。所有 residual 使用旧值，不从 reference/prediction 重新
覆盖；随后断言两种表达在 float64 容差内一致。

Run:

```bash
python -m pytest \
  Uncertainty_Quantification/LLPR/.local_adaptation/test_adapt_results.py -v
```

Expected: PASS。

- [ ] **Step 5: 检查转换代码没有模型依赖**

Run:

```bash
rg -n "checkpoint|torch.load.*model|mace\\.|run_build|run_calibrate|autograd" \
  Uncertainty_Quantification/LLPR/.local_adaptation/adapt_results.py
```

Expected: 除禁止词检查本身外无匹配；程序只允许 `torch.load` 六个结果 PT。

## Task 3: 全量 staging 转换和跨 variant 验证

**Files:**
- Temporary modify: `Uncertainty_Quantification/LLPR/.local_adaptation/adapt_results.py`
- Write staging CSV/JSON only.
- Update untracked audit report.

**Interfaces:**
- Consumes: 六个完整 PT。
- Produces: staging 下的三组 CSV/summary。

- [ ] **Step 1: 建立 canonical structure ID**

逐行要求：

```python
assert he["structure_index"] == index
assert hef["structure_index"] == index
assert he["sample_id"] == hef["sample_id"]
assert he["natoms"] == hef["natoms"] == hf["natoms"]
canonical_id = str(he["sample_id"])
```

Hf 通过总 Energy、预测 Energy 和原子数与 He/Hef 交叉验证后使用 canonical ID。

- [ ] **Step 2: 写三个 variant 的规范 CSV**

使用正式 `ENERGY_FIELDS`、`FORCE_FIELDS`、`FORCE_STRUCTURE_FIELDS` 和可精确
回读 float64 的 `csv.DictWriter`。逐结构写 Force-structure 汇总，不把
447,963 个 Force 分量同时展开保存在额外内存。

- [ ] **Step 3: 恢复六个 alpha 并写 summary**

对所有有限 `q>0` 的记录恢复 alpha；要求各路径相对偏差不超过 `1e-10`。
summary 使用正式固定 schema，ridge 为 `1e-12`，不存在的 Cholesky 诊断字段
写 `null`，不得编造数值。

- [ ] **Step 4: 核对全量数量**

断言每个 variant：

```python
assert energy_rows == 19_374
assert force_component_rows == 447_963
assert force_structure_rows == 19_374
```

断言三个 variant 的 Energy 键和 Force `(structure_id, atom_index, direction)`
完全一致。

- [ ] **Step 5: 写 staging audit**

记录字段映射、Hf 可逆单位检查最大误差、alpha、行数、跨 variant 对齐和 staging
文件 SHA256。状态更新为 `staging_complete`。

## Task 4: 使用正式验证/绘图发布并删除一次性代码

**Files:**
- Consume: staging 统一结果
- Create: 正式 `evaluation/deterministic/`
- Create: 新 PNG/PDF/statistics/manifest
- Delete temporary: `Uncertainty_Quantification/LLPR/.local_adaptation/`
- Keep untracked: audit directory outside publication root

**Interfaces:**
- Consumes: 正式 `validate_publication_root`、`run_plot`。
- Produces: 完整旧结果适配发布包，无迁移代码。

- [ ] **Step 1: 用正式 validator 验证 staging**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr validate \
  --result-root <staging-path>
```

Expected: `status=valid`；不是调用一次性程序自己的替代验证。

- [ ] **Step 2: 原子发布并生成全新图**

把通过验证的 staging 原子重命名为 `evaluation/deterministic/`，然后运行：

```bash
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --result-root <published-path> \
  --config Uncertainty_Quantification/LLPR/configs/plot_publication.yaml
```

Expected: 只产生新 PNG/PDF，不复制旧图。

- [ ] **Step 3: 重新验证源 SHA256 和旧 Git 状态**

重新计算六个旧 PT SHA256，与 Task 1 manifest 逐字相等。比较旧仓库
`git status --short` 与 Task 1 快照，确保本任务没有新增修改。

- [ ] **Step 4: 删除一次性转换代码和临时测试**

删除整个：

```text
Uncertainty_Quantification/LLPR/.local_adaptation/
```

确认：

```bash
rg -n "UQ_LLPR|llpr_.*prediction_details|per_component_force_true|migration|迁移" \
  Uncertainty_Quantification/LLPR/llpr \
  Uncertainty_Quantification/LLPR/configs \
  Uncertainty_Quantification/LLPR/README.md
```

Expected: 无匹配。

- [ ] **Step 5: 最终验收**

Run:

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests -v
python -m Uncertainty_Quantification.LLPR.llpr validate \
  --result-root <published-path>
git status --short
```

Expected:

- 正式测试全部 PASS。
- 适配发布包 valid。
- 每种 variant 为 19,374 / 447,963 / 19,374 行。
- 新图和 plotting manifest 完整。
- Git 中没有一次性迁移代码。
- 只提交必要的正式代码修订；适配结果和本地审计按 outputs 策略保留。
