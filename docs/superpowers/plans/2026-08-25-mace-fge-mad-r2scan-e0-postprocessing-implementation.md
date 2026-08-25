# MACE FGE MAD-r2SCAN E0 双方法后处理实施计划

> **执行要求：** 实施本计划时使用 superpowers:subagent-driven-development；若改为独立执行会话，则使用 superpowers:executing-plans。每个任务都按测试先行、实现、远端验证、审查的顺序完成。

**目标：** 在不修改 checkpoint、不重跑 MAD-r2SCAN test 模型前向、且不改变原始 FGE prediction 的前提下，实现 direct_test_e0 与 model_aware_val_e0 两套 E0 后处理，复用现有 UQ/评估/绘图模块，最终在远端生成每种方法 43 PNG + 43 PDF，并只将图片拉取到本地。

**架构：** 公开 FGE 模块只补齐通用的 calc-first extxyz 读取和可选的评估/绘图上下文；所有 E0 专用读取、校准、修正、artifact 和远端入口放在 Uncertainty_Quantification/FGE/postprocessing/e0_correction/，保持在 publication_files.txt 之外。修正后的 prediction shard 继续严格使用 fge.prediction.v1；方法、校准和完整性信息写入旁车文件。方法一从 test 的 energy 与 atomization_energy 构造逐结构 MAD 基线，方法二只在 val 上逐 member 求解最小范数 Delta_E0。

**技术栈：** Python 3.10+、PyTorch float64、NumPy、ASE、MACE、Matplotlib、SciPy、PyYAML、pytest、W&B、Slurm。

---

## 全局执行边界

- 本地只编辑、审查和提交源码、测试、配置、Slurm 与文档；所有 pytest、smoke、模型加载、校准、评估和绘图均在远端执行。
- 优先使用远端已有 Python，并在会话中设置任务专用变量：

      FGE_REMOTE_PYTHON=/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/mace/bin/python

  先验证环境并在仓库根执行 python -m pip install -e .。不得克隆 conda 环境。只有该环境确实不可用时，才创建全新的 mace_new 环境并执行 pip install -e .。
- 远端仓库固定为：

      /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/mace_new

- test 原始 prediction 全程只读，不允许调用任何 test model forward。唯一允许的新模型计算是方法二的 val-only energy forward，且必须同时使用 compute_force=False、compute_stress=False。
- val-only forward 前只依据结构 ID/无标签结构指纹，从 val 整结构排除与 test 重叠的校准结构；test 结构、顺序和现有 raw prediction 不变。当前已审计输入应排除 2 条，数量不符硬失败。
- 原始 prediction、evaluation、正式训练结果和 checkpoint 不覆盖、不移动、不改名；运行前后对这些文件做 SHA-256 审计。
- corrected .pt 严格保持 fge.prediction.v1 exact-key schema。任何 E0、method、calibration 或来源字段都不得写入该 payload。
- corrected manifest 只记录方法、逻辑数据集、member、observable、shape、范围和校准摘要，不绑定旧仓库、旧路径或旧来源身份。远端内部 integrity audit 可记录 raw/extxyz/checkpoint SHA，但不发布、不拉回本地。
- 两种方法只处理 Energy 和 Force，observables 必须精确等于 ["energy", "forces"]，不得生成 Stress。
- 质量退化、低相关性、可接受秩亏和 W&B 自身失败只产生 warning。字段缺失、非有限值、对齐失败、不可识别校准、hash/schema 破坏、覆盖冲突、图合同失败等硬合同问题抛 HardFailure 并使作业非零退出。
- W&B 功能保留，但只记录阶段状态、计数、耗时、warning 数和有限标量指标；不上传 prediction、校准数组、图片、日志、checkpoint 或其他 artifact。
- 最终只拉取 PNG/PDF 到：

      Uncertainty_Quantification/Plots/FGE/mad_r2scan_e0/direct_test_e0/
      Uncertainty_Quantification/Plots/FGE/mad_r2scan_e0/model_aware_val_e0/

---

### Task 1：修复标准 extxyz 的 calc-first 读取

**Files:**

- Modify: Uncertainty_Quantification/FGE/fge/extxyz_standard.py
- Modify: Uncertainty_Quantification/FGE/tests/test_extxyz_standard.py

- [ ] **Step 1：写 calc-first 优先级失败测试**

  覆盖 energy、forces、stress 同时存在于 atoms.calc.results 和 atoms.info/arrays 时必须优先 calculator；calculator 缺值或值为 None 时回退现有 ase_value 行为；atomization_energy 从标量字段读取。

- [ ] **Step 2：在远端确认 RED**

  Run:

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/tests/test_extxyz_standard.py

  Expected: 新优先级测试失败，现有回退测试继续通过。

- [ ] **Step 3：实现窄 helper**

  在 extxyz_standard.py 增加私有 _calculator_result(atoms, key)，仅在 atoms.calc.results 为 Mapping 且 key 存在且非 None 时返回该值。read_energy、read_forces、read_stress 先调用该 helper，再回退 ase_value。增加 read_atomization_energy(atoms, key="atomization_energy")，不得把它解释为全电子总能量。

- [ ] **Step 4：增加错误测试**

  覆盖 calculator 非 mapping、缺字段、NaN/Inf 交由上层合同拒绝、custom key 回退，以及 stale info/arrays 不得覆盖 calculator。

- [ ] **Step 5：在远端确认 GREEN**

  Run:

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/tests/test_extxyz_standard.py \
        Uncertainty_Quantification/FGE/tests/test_data.py

  Expected: PASS，且既有 custom extxyz 行为不变。

- [ ] **Step 6：提交**

      git commit -m "fix(fge): prefer ASE calculator reference values"

---

### Task 2：建立非发布 E0 合同、配置与数据模型

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/__init__.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/__init__.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/models.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_models.py

- [ ] **Step 1：写严格方法与配置失败测试**

  固定仅接受 direct_test_e0、model_aware_val_e0；四个实验标签必须连续、唯一且与四份配置一一对应；数据字段必须显式包含 energy、forces、atomization_energy 和 head；拒绝未知键、重复方法、不安全逻辑标签、test/val 同一逻辑 split、启用 stress 或允许 test forward。

- [ ] **Step 2：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_models.py

  Expected: FAIL because models module does not exist.

- [ ] **Step 3：实现不可变合同**

  建议数据模型：

      CorrectionMethod
      E0DatasetConfig
      ExperimentConfig
      E0RunConfig
      CalibrationFit
      CorrectionSummary
      WarningRecord

  CorrectionMethod 同时固定 machine name、显示标题、calibration_split、uses_test_reference_labels 和 evaluation_role：

      direct_test_e0:
          title = Direct test-informed E0
          calibration_split = test
          uses_test_reference_labels = true
          evaluation_role = transductive_diagnostic

      model_aware_val_e0:
          title = Model-aware val-calibrated E0
          calibration_split = val
          uses_test_reference_labels = false
          evaluation_role = calibrated_test

- [ ] **Step 4：定义 schema 与来源中立字段**

  在 models.py 集中定义 calibration、correction、integrity schema version 与 exact-key validators。published correction manifest 明确拒绝绝对路径、raw/checkpoint 路径、旧仓库名和输入内容 SHA；internal integrity audit 使用独立 schema，不能由发布 reader 返回。

- [ ] **Step 5：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_models.py

  Expected: PASS。

- [ ] **Step 6：提交**

      git commit -m "feat(fge): define E0 postprocessing contracts"

---

### Task 3：实现 MAD 标签、过滤、组成矩阵与 split 对齐

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/labels.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_labels.py

- [ ] **Step 1：写 whole-structure filtering 测试**

  构造含支持与不支持元素的 extxyz，断言只能整条删除结构，不能删除单个原子；保留稳定 source index；过滤结果顺序与输入顺序一致。

- [ ] **Step 2：写字段与 calculator 读取测试**

  test 必须读取 calculator-first energy/forces 和 atomization_energy；val 必须读取 calculator-first energy/forces，但方法二不得读取 val atomization_energy 作为拟合目标。缺字段、shape 错误、NaN/Inf、空结构均硬失败。

- [ ] **Step 3：写 prediction shard 对齐测试**

  用多 shard fixture 核对每 shard 的 structure range、atom range、n_atoms、structure_ptr、总 S/A 和 extxyz 流式消费完全一致。结构顺序、过滤闭包或原子计数任一不一致都在生成输出前失败。

- [ ] **Step 4：写 split 去重与 val 去污染测试**

  优先使用 configuration_id；ID 缺失时，对原子序数 int64、位置与 cell 的 little-endian float64 原始字节、pbc uint8 及明确的 shape 分隔符计算 SHA-256。指纹不包含 energy、forces、atomization_energy、路径或 source index，不做可能合并不同结构的十进制舍入。

  任一 split 内重复结构硬失败。跨 split 重复按固定规则 exclude_test_identity_overlap_from_val 从 val 整结构排除，test 保持原顺序和数量；排除发生在 val forward 前。当前固定数据必须排除 2 条，否则按输入漂移硬失败。改变 test energy、forces 或 atomization_energy 不得改变排除集合。

- [ ] **Step 5：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_labels.py

  Expected: FAIL because labels module does not exist.

- [ ] **Step 6：实现流式 reader**

  提供：

      supported_atomic_numbers(...)
      iter_filtered_structures(...)
      structure_identity(...)
      build_composition_matrix(...)
      iter_aligned_test_label_shards(...)
      load_validation_labels(...)
      decontaminate_validation_split(...)
      assert_disjoint_splits(...)

  composition 列顺序由已审计 checkpoint atomic_numbers 显式传入，结果为有限 float64 非负整数矩阵。test shard 同时返回：

      energy_reference
      atomization_energy
      forces_reference
      composition
      n_atoms
      structure_ptr
      structure_ids

- [ ] **Step 7：增加全零 reference 防回归**

  当 extxyz reader 已读到非零有限标签，但生成的完整 corrected energy_reference 或 forces_reference 仍全零时硬失败。合法的局部零值不得误报。

- [ ] **Step 8：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_labels.py \
        Uncertainty_Quantification/FGE/tests/test_extxyz_standard.py

  Expected: PASS。

- [ ] **Step 9：提交**

      git commit -m "feat(fge): add aligned MAD E0 label loading"

---

### Task 4：实现两种 E0 数值算法与可识别性检查

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/algorithms.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_algorithms.py

- [ ] **Step 1：写方法一公式失败测试**

  对多个 composition 和不同 member E0 验证：

      mad_baseline = reference_total - atomization_energy
      model_baseline = composition @ model_e0_member
      corrected = raw_total - model_baseline + mad_baseline

  atomization_energy 不得被当作全电子总能量，也不得在 val 拟合。

- [ ] **Step 2：写方法二逐 member 失败测试**

  对每个 member 独立构造已知 Delta_E0，验证：

      b_val = reference_val - raw_val_member
      delta = numpy.linalg.lstsq(A_val, b_val, rcond=None)
      corrected_test = raw_test_member + A_test @ delta

  改变 test reference 后 delta 必须完全不变，证明 test 未参与拟合。

- [ ] **Step 3：写可识别性失败测试**

  用 numpy.linalg.svd(A_val, full_matrices=True) 检查 rank deficiency。固定 rank_tol = eps_float64 * max(A_val.shape) * max(singular_values[0], 1.0)，rank 为大于 rank_tol 的奇异值数，V_null = Vh[rank:].T；固定 impact_tol = rank_tol * max(1.0, norm(A_test, ord=2))。若 max(abs(A_test @ V_null)) > impact_tol，硬失败；若秩亏不影响 A_test @ Delta，则保存 minimum-norm 解并产生 warning。test 包含 val 未覆盖元素必须硬失败。

- [ ] **Step 4：写 hard/warning 边界测试**

  非有限值、负 composition、shape 错误、元素列错序、空输入、E0 缺元素和不可识别 correction 为 HardFailure；大 calibration residual、病态但仍可识别的矩阵和较弱 RMSE 只返回 WarningRecord。

- [ ] **Step 5：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_algorithms.py

  Expected: FAIL because algorithms module does not exist.

- [ ] **Step 6：实现纯 float64 内核**

  建议入口：

      direct_test_correction(...)
      fit_member_delta_e0(...)
      apply_member_delta_e0(...)
      test_space_identifiability(...)
      correction_diagnostics(...)

  所有输入在计算前统一为 CPU float64 NumPy array，输出包含 rank、singular_values、condition_number、residual RMSE/max、nullspace test impact 与 rcond=None 证据。

- [ ] **Step 7：验证 UQ 不变量**

  测试共同平移时 energy STD/GMD 不变；member E0 不同时重算后的 STD/GMD 与手算一致；两种方法均不改变 force members、mapping 和 n_atoms。

- [ ] **Step 8：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_algorithms.py

  Expected: PASS。

- [ ] **Step 9：提交**

      git commit -m "feat(fge): add dual E0 correction algorithms"

---

### Task 5：提取 member checkpoint E0 并增加 val-only energy inference

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/inference.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_inference.py

- [ ] **Step 1：写 checkpoint E0 提取失败测试**

  mock 单 head、多 head、atomic_energies 一维/二维布局，验证按 config 的 head_name 与 model.atomic_numbers 提取每个 member 实际 E0。缺 head、列错位、非有限值、缺 test 元素、无法证明 composition-only E0 时硬失败。

- [ ] **Step 2：写正式 manifest 只读测试**

  必须复用 load_training_manifest 校验 validation.json、result_manifest.json、member 顺序与 raw checkpoint SHA。禁止直接 glob checkpoint，禁止使用 base model E0 代替 member E0。

- [ ] **Step 3：写 val-only 调用合同测试**

  mock model 记录 forward kwargs，精确断言：

      training = false
      compute_force = false
      compute_virials = false
      compute_stress = false

  同时断言方法二仅接受 split=val；任何 split=test 调用在 model forward 前硬失败。

- [ ] **Step 4：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_inference.py

  Expected: FAIL because inference module does not exist.

- [ ] **Step 5：实现 member preflight**

  一次性审计四实验 × 8 members 的 member IDs、checkpoint hash、head、atomic_numbers、E0、dtype 和 r_max；在任何 corrected/calibration 输出前完成。共同支持元素集合必须与既有 test filtering 闭包一致。

- [ ] **Step 6：实现 energy-only loader 与 forward**

  复用 _load_model、build_mace_loaders、default_dtype 和正式 member 顺序，但新建专用 infer_validation_energies，不调用 prediction._infer_one，也不生成 forces/stress。输出为独立 calibration schema：

      split = val
      member_ids
      structure_ids
      atomic_numbers
      composition
      energy_members
      energy_reference
      n_atoms

- [ ] **Step 7：增加 dtype/device 与无梯度回归测试**

  验证每个模型使用自身 parameter dtype，模型输出转 CPU float64，参数 requires_grad 状态不泄漏到后续 member，并且 test forward 计数始终为零。

- [ ] **Step 8：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_inference.py \
        Uncertainty_Quantification/FGE/tests/test_prediction.py

  Expected: PASS，现有 prediction 行为不变。

- [ ] **Step 9：提交**

      git commit -m "feat(fge): add validation-only E0 calibration inference"

---

### Task 6：实现 corrected shard、calibration 与完整性 artifact

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/artifacts.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_artifacts.py

- [ ] **Step 1：写 fixed-schema 失败测试**

  从 canonical raw payload 派生 corrected payload，断言 exact keys 仍为 fge.prediction.v1，只允许替换 energy_members、energy_reference、forces_reference；forces_members、n_atoms、atom_to_structure、structure_ptr、member_ids 和 observables 必须 torch.equal。

- [ ] **Step 2：写 method-scoped layout 测试**

  固定布局：

      outputs/e0_correction/direct_test_e0/inference/EXPERIMENT/mad_r2scan_test/
      outputs/e0_correction/model_aware_val_e0/calibration/EXPERIMENT/
      outputs/e0_correction/model_aware_val_e0/inference/EXPERIMENT/mad_r2scan_test/

  两种方法不能共享 corrected shard 目录；raw layout 不能位于 corrected root 内。

- [ ] **Step 3：写来源中立与 internal audit 隔离测试**

  correction_manifest.json 只含 logical method/split/member/observable/shape/range/calibration summary/warnings，并记录 validation_decontamination=exclude_test_identity_overlap_from_val 与 overlap_removed_count=2；不记录具体 configuration_id。拒绝绝对路径、旧 repo 文本和 raw/checkpoint/extxyz SHA。internal_integrity_audit.json 单独保存被排除 ID、输入/输出 hash、远端物理路径和运行时证据，且发布 reader 不返回它。

- [ ] **Step 4：写可恢复与防覆盖测试**

  同一 method-scoped signature 可验证并复用完整 shard；半写 artifact、hash 不一致、calibration summary 不一致或现有不同 signature 均硬失败。PASS manifest 最后原子发布；失败时不得出现 PASS。

- [ ] **Step 5：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_artifacts.py

  Expected: FAIL because artifacts module does not exist.

- [ ] **Step 6：复用既有 artifact helpers**

  复用 DerivedLayout、build_prediction_shard_signature、write_prediction_shard、verify_prediction_shard_if_present、validate_prediction_payload、atomic_write_json、atomic_torch_save 和 sha256_file。不要复制 canonical validator。

- [ ] **Step 7：实现 calibration artifacts**

  model_aware_val_e0 对每个 experiment 原子发布 val energy prediction、每 member delta_e0、rank/SVD/残差/可识别性诊断和 calibration_audit.json。direct_test_e0 不创建伪 val calibration，只保存逐结构 MAD baseline 摘要。

- [ ] **Step 8：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_artifacts.py \
        Uncertainty_Quantification/FGE/tests/test_dataset_prediction.py

  Expected: PASS。

- [ ] **Step 9：提交**

      git commit -m "feat(fge): add E0 correction artifacts"

---

### Task 7：组装校准、应用、评估工作流

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/workflow.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_workflow.py

- [ ] **Step 1：写端到端 synthetic 双方法失败测试**

  使用 2 members、2 shards、多元素的 canonical fixture，验证 direct_test_e0 与 model_aware_val_e0 均生成 evaluator 可读取的 corrected layout；test raw prediction 不发生 forward 或写入。

- [ ] **Step 2：写阶段顺序与硬失败短路测试**

  固定顺序：

      preflight
      -> label alignment and within-split uniqueness
      -> remove test-identity overlaps from val
      -> verify split disjointness
      -> checkpoint E0 extraction
      -> val energy-only prediction when required
      -> calibration
      -> corrected shard application
      -> evaluation
      -> plot input publication

  任一硬失败后不得执行后续阶段，也不得留下 PASS manifest。

- [ ] **Step 3：写 force/reference 不变量测试**

  两种方法的 forces_members 必须与 raw 逐位相同；两种方法之间也必须相同。forces_reference 从正确读取的 extxyz 重建，两方法一致。正式 MatPES validation 权重从 training manifest 复用，不得由 MAD val/test 重算。

- [ ] **Step 4：写 val 去污染与标签防泄漏测试**

  断言 overlap val structures 在任何 model forward 前被删除，test raw structure count/order 不变，公开 manifest 只含规则和 count=2，internal audit 才含具体 ID。改变 test labels 后 val selection、val energy prediction、Delta_E0 与 corrected test energy 均不变。

- [ ] **Step 5：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_workflow.py

  Expected: FAIL because workflow module does not exist.

- [ ] **Step 6：实现公开 orchestration**

  建议入口：

      calibrate_experiment(...)
      apply_correction(...)
      evaluate_corrected_experiment(...)
      run_experiment(...)
      run_all_experiments(...)

  四实验和两方法由严格 config 驱动；算法模块不得硬编码远端路径或旧来源。

- [ ] **Step 7：实现 raw snapshot 审计**

  在 preflight 记录正式 marker、training manifest、checkpoint、raw prediction shard/manifest hash；结束后重新计算并逐项比较。任何变化硬失败。该证据只写 internal integrity audit。

- [ ] **Step 8：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_workflow.py \
        Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py

  Expected: PASS。

- [ ] **Step 9：提交**

      git commit -m "feat(fge): orchestrate dual MAD E0 experiments"

---

### Task 8：给评估报告和 43 图增加可审计方法标签

**Files:**

- Modify: Uncertainty_Quantification/FGE/fge/dataset_evaluation.py
- Modify: Uncertainty_Quantification/FGE/fge/plot_single.py
- Modify: Uncertainty_Quantification/FGE/fge/plot_compare.py
- Modify: Uncertainty_Quantification/FGE/fge/plot_workflow.py
- Modify: Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py
- Modify: Uncertainty_Quantification/FGE/tests/test_plot_single.py
- Modify: Uncertainty_Quantification/FGE/tests/test_plot_compare.py
- Modify: Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py

- [ ] **Step 1：写向后兼容 API 失败测试**

  现有调用不传 context 时标题、audit schema 和 43/59 图合同不变。新增可选 keyword-only 参数：

      evaluate_dataset(..., report_context=None)
      render_single_run(..., title_context=None)
      render_comparisons(..., title_context=None)
      render_dataset_figures(..., title_context=None, required_marker=None)

- [ ] **Step 2：写方法标题失败测试**

  direct_test_e0 的 report、evaluation audit、43 个实际图标题和 plot_audit 必须包含 Direct test-informed E0；model_aware_val_e0 必须包含 Model-aware val-calibrated E0。比较图也必须有方法级 suptitle。

- [ ] **Step 3：写 required marker 硬失败测试**

  wrapper 传 required_marker=test-informed 或 val-calibrated 时，若 report、任一 FigureRecord.title 或 plot audit 缺失 marker，必须在原子发布 final figure directory 前硬失败。

- [ ] **Step 4：在远端确认 RED**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py \
        Uncertainty_Quantification/FGE/tests/test_plot_single.py \
        Uncertainty_Quantification/FGE/tests/test_plot_compare.py \
        Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py

  Expected: 新 context 测试失败。

- [ ] **Step 5：实现中央标题装饰**

  避免在 8 个绘图路径重复拼接。FigureRecord 末尾增加 title: str | None = None，保持旧构造兼容；plot audit 增加 analysis_context 和每图 title。比较图使用 figure.suptitle，单图在现有标题后追加统一 context。

- [ ] **Step 6：保持 43 图无 Stress 合同**

  两种方法各自必须为 4 experiments × 2 branches × 5 + 3 comparisons = 43 logical figures，同时得到 43 PNG 和 43 PDF。新增 context 不改变文件名或 plot data 数值。

- [ ] **Step 7：在远端确认 GREEN**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/tests/test_dataset_evaluation.py \
        Uncertainty_Quantification/FGE/tests/test_plot_single.py \
        Uncertainty_Quantification/FGE/tests/test_plot_compare.py \
        Uncertainty_Quantification/FGE/tests/test_dataset_plotting.py \
        Uncertainty_Quantification/FGE/tests/test_plot_workflow.py

  Expected: PASS。

- [ ] **Step 8：提交**

      git commit -m "feat(fge): label E0 evaluation and plots"

---

### Task 9：增加脚本、配置、W&B、Slurm、说明和 Git 边界

**Files:**

- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/__init__.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/calibrate_e0.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/apply_e0.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/evaluate_corrected.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/run_pipeline.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/scripts/submit_mad_r2scan_e0.slurm
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/configs/mad_r2scan.yaml
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/README_zh.md
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_scripts.py
- Create: Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_remote_entry.py
- Modify: Uncertainty_Quantification/FGE/tests/test_publication_boundary.py
- Modify: .gitignore

- [ ] **Step 1：写 CLI 与 config 失败测试**

  所有输入路径、逻辑 label、method、output root、batch/shard size 和 W&B mode 必须显式可审计。默认 config 引用：

      val: data/dataset/val/co/co_0.extxyz
      test: data/dataset/DS_7q71mf99le0c_0/co/co_0.extxyz
      dataset label: mad_r2scan_test
      batch size: 8
      shard size: 256

  四实验配置固定为已确认的四份 FGE GPU YAML，但核心模块不硬编码它们。

- [ ] **Step 2：实现脚本薄入口**

  calibrate_e0 只允许 model_aware_val_e0；apply_e0 接受两个方法但不执行 evaluator；evaluate_corrected 只消费 PASS corrected manifest；run_pipeline 固定阶段顺序并支持从已验证阶段幂等恢复。脚本不得复制算法。

- [ ] **Step 3：实现有限 W&B 记录**

  复用 fge.wandb.create_wandb_logger。记录 method、experiment、stage、K/S/A、shard_count、duration、warning_count 和有限 RMSE/correlation 标量；禁止调用 artifact upload。online/offline/disabled 可配置，W&B 失败降级 warning 和 ignored fallback JSONL，不改变科学计算 PASS/FAIL。

- [ ] **Step 4：写 Slurm 入口**

  以已验证参数为起点：

      partition=ai
      gres=gpu:a800:1
      cpus-per-task=8
      time=10:00:00

  脚本使用 set -euo pipefail、显式仓库根、FGE_REMOTE_PYTHON、OMP_NUM_THREADS=8、MKL_NUM_THREADS=8 和 PYTHONUNBUFFERED=1。提交前用 scontrol 核对 ai partition 的时限；若 10 小时超限，调整为该 partition 允许值并记录。日志写远端非发布目录，不进入 Git。

- [ ] **Step 5：补充中文说明**

  README_zh.md 说明两个公式、test-informed 限制、val-only 前向、hard/warning 边界、输出树、W&B 边界、smoke/全量命令、Slurm 状态判断和只拉图片规则。

- [ ] **Step 6：固定 publication 与 Git ignore**

  publication_files.txt 保持不变。测试明确断言 postprocessing/ 不在 allowlist，同时 fge/、scripts/、tests/ 仍可发布。根 .gitignore 增加：

      /Uncertainty_Quantification/Plots/FGE/**

  避免 PDF 被误提交；不得放行 generated plot 子目录。

- [ ] **Step 7：在远端运行脚本测试**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_scripts.py \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests/test_remote_entry.py \
        Uncertainty_Quantification/FGE/tests/test_publication_boundary.py \
        Uncertainty_Quantification/FGE/tests/test_wandb.py

  Expected: PASS。

- [ ] **Step 8：提交**

      git commit -m "feat(fge): add remote E0 postprocessing pipeline"

---

### Task 10：远端安装当前代码并运行完整测试

**Files:** No new production files unless a test exposes a defect.

- [ ] **Step 1：通过当前已登录 MobaXterm 会话进入远端仓库**

  核对 hostname、用户、仓库绝对路径、当前分支和待测 commit SHA。只同步当前分支的已提交源码，不复制本地 outputs、Plots、data、checkpoint 或环境目录。

- [ ] **Step 2：验证现有 mace 环境**

      $FGE_REMOTE_PYTHON -c "import ase, mace, numpy, scipy, torch, yaml; print(torch.__version__)"
      $FGE_REMOTE_PYTHON -m pip install -e .

  若失败，先诊断依赖/仓库问题。只有确认环境本身不可用时才新建 mace_new；禁止 conda clone 或环境目录复制。

- [ ] **Step 3：运行新增核心测试**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction/tests

  Expected: PASS。

- [ ] **Step 4：运行 FGE 全套回归**

      $FGE_REMOTE_PYTHON -m pytest -q \
        Uncertainty_Quantification/FGE/tests

  Expected: PASS。任何失败先按 superpowers:systematic-debugging 定位根因，不得通过放宽 hard contract 或跳过测试掩盖。

- [ ] **Step 5：运行静态占位符检查**

      rg -n "TODO|FIXME|NotImplemented|placeholder|旧来源|mace/Ensemble" \
        Uncertainty_Quantification/FGE/postprocessing/e0_correction \
        Uncertainty_Quantification/FGE/fge

  Expected: 无未解释占位符；运行代码与 published manifest 不绑定旧来源。

- [ ] **Step 6：记录测试证据**

  在远端 ignored work 状态中记录 commit SHA、Python/Torch/MACE 版本、测试数、耗时和 PASS；不把 pytest cache 或日志拉回本地。

---

### Task 11：远端 CPU 小数据双方法闭环

**Files:** 只生成远端 ignored smoke artifacts.

- [ ] **Step 1：建立只读输入快照**

  验证四组 raw prediction manifest 均为 PASS，并从 manifest 动态读取 K/S/A/shard_count。预期正式 test 为 K=8、S=16072、A=311657、63 shards，但程序不得硬编码这些数字。记录原始 manifest/shard/checkpoint hash。

- [ ] **Step 2：构造确定性 smoke 子集**

  从现有 test raw shards 只读切片 8–16 个结构，不做 test forward；从 val 取不与 test 重复且覆盖所需元素的固定小集合。选择至少提供两个正有限 residual/UQ 点，满足 log 图合同。保存 source index 只在 internal audit。

- [ ] **Step 3：执行真实 checkpoint preflight**

  在 CPU 上加载四实验全部 raw member，核对 head、atomic_numbers、member E0 与支持元素闭包。任何成员不一致立即硬失败。

- [ ] **Step 4：执行方法二 val-only energy forward**

  使用真实 checkpoint、compute_force=False、compute_stress=False。审计输出不得出现 forces 或 stress 字段，并证明 test forward count=0。

- [ ] **Step 5：执行两种修正与既有 evaluator**

  每种方法 × 四实验生成 corrected shards，运行 equal_weight 与原 validation_weighted UQ/metrics/risk-coverage。断言 raw hash 不变、force members 逐位不变、方法一公式和方法二 Delta_E0 可独立重算。

- [ ] **Step 6：生成两套 smoke 图**

  每种方法生成 43 PNG + 43 PDF；标题和 plot audit 带对应 method context；无 stress 文件；所有文件非空。重复运行相同签名应幂等复用或明确拒绝覆盖，不能静默重算出不同 artifact。

- [ ] **Step 7：比较格式合同**

  对 smoke 与正式目标使用同一 schema validator，证明 corrected prediction、evaluation、correlation/risk CSV 和 plot audit 的字段/层级一致；仅 shape/count 随数据规模变化。

---

### Task 12：通过 MobaXterm 提交并完成远端全量作业

**Files:** 只生成远端 ignored full-run artifacts.

- [ ] **Step 1：全量 preflight**

  核对 test 原始四组 prediction 都满足：

      K = 8
      S = 16072
      A = 311657
      shard_count = 63
      observables = ["energy", "forces"]

  val/test 原始结构数分别从实际 extxyz 读取；支持元素过滤后数量由程序计算，不硬编码。先核对每个 split 内唯一，再从 val 排除与 test 身份重叠的 2 个结构并验证无残留；test 数量、顺序和 raw prediction 均不得改变。随后完成支持元素、head、E0、reference 和 hash 审计。

- [ ] **Step 2：提交唯一 Slurm job**

  在当前已登录 MobaXterm 会话运行 sbatch --parsable，立即保存 job ID。随后运行 squeue -j JOB_ID 验证；若任务瞬时离队，使用 sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed 判断。不得因 squeue 暂时为空而盲目重复提交。

- [ ] **Step 3：监控到终态**

  定期查看 squeue/sacct 和该 job 的 stdout/stderr；PENDING/RUNNING 时继续等待。FAILED、CANCELLED、TIMEOUT、OUT_OF_MEMORY 或非零 ExitCode 视为实验失败，先报告与修复，不把部分输出标成 PASS。

- [ ] **Step 4：执行完整计算链**

      four-experiment preflight
      -> four-experiment val-only energy prediction
      -> per-member model-aware calibration
      -> direct test baseline construction
      -> two methods x four experiments corrected shards
      -> two methods x four experiments evaluation/UQ/risk
      -> two method-scoped 43-figure suites

  direct_test_e0 不运行 val 拟合；model_aware_val_e0 不读取 test energy/atomization 参与 Delta_E0 拟合；两者都不运行 test model forward。

- [ ] **Step 5：审计 hard/warning**

  hard contract 失败必须让 Slurm job 非零退出。质量 warning 保留在 correction/evaluation/plot audit 和 W&B 有限标量中，但不阻断结果。W&B 上传列表必须为空。

- [ ] **Step 6：核对全量产物**

  每种方法、每实验：corrected prediction PASS、evaluation PASS、K/S/A 与 raw 一致、63 corrected shards、两权重分支完整。每种方法：43 PNG + 43 PDF + plot audit，标题正确，无 Stress。

- [ ] **Step 7：复核输入不变**

  比较运行前后正式训练 marker、checkpoint、raw prediction 与旧 evaluation hash。任何变化硬失败；internal integrity audit 可保存证据但不回传。

---

### Task 13：只拉图片、本地复核、代码审查与最终提交审计

**Files:**

- Generated and ignored only:
  - Uncertainty_Quantification/Plots/FGE/mad_r2scan_e0/direct_test_e0/
  - Uncertainty_Quantification/Plots/FGE/mad_r2scan_e0/model_aware_val_e0/

- [ ] **Step 1：远端最终状态确认**

  用 sacct 确认主 job COMPLETED、ExitCode=0:0；检查 stdout/stderr 没有未处理 traceback；两个 method 的 correction/evaluation/plot audits 均 PASS。

- [ ] **Step 2：仅回传 PNG/PDF**

  通过当前 MobaXterm 连接只复制两个 method figure root 下的 .png 与 .pdf。不得回传 prediction、evaluation、PT、JSON、CSV、calibration、integrity audit、W&B、日志、checkpoint 或 data。

- [ ] **Step 3：本地文件合同复核**

  每种方法精确 43 PNG + 43 PDF，PNG/PDF stem 集合一致、文件均非空；两方法合计 86 PNG + 86 PDF。使用远端即时 hash 列表核对本地文件内容，但不把 hash 清单保存进发布结果目录。

- [ ] **Step 4：人工抽查图片**

  每种方法至少检查 Energy parity、Force parity、Energy uncertainty-residual、risk-coverage 和三类 comparison；确认标题分别明确显示 test-informed 或 val-calibrated，文字不重叠、轴单位正确、图像非空。

- [ ] **Step 5：规格审查**

  使用 superpowers:requesting-code-review，逐项核对：

      方法一只用 test energy - atomization_energy
      方法二只用 val total energy 和 raw val member energy
      sign 为 reference - prediction 后加 A @ Delta
      每 member 独立拟合
      test forward 为零
      forces/mapping/raw/checkpoint 不变
      corrected .pt exact schema
      manifest 来源中立
      hard/warning 边界
      43 图与方法标签

- [ ] **Step 6：完成前验证**

  使用 superpowers:verification-before-completion，重新核对远端完整 pytest、smoke、Slurm 终态、全量 artifact 数、raw hash 和本地图片数。不得以历史输出或口头判断代替当前证据。

- [ ] **Step 7：核对当前分支提交边界**

      git status --short --branch
      git log --oneline --decorate -20

  只提交本计划实施产生的 FGE 源码、测试、配置、Slurm、文档和 .gitignore；Plots 目录应被忽略。不得加入 BootStrapping、LLPR、ConfidenceHead、CARNet、outputs 或其他并行改动。

- [ ] **Step 8：提交最终实施**

  若前面按 task 分提交，最后只提交必要的审查修复；若采用压缩提交，使用：

      git commit -m "feat(fge): add MAD r2SCAN E0 postprocessing"

  提交后再次记录 branch、HEAD SHA、远端测试证据与未跟踪的用户文件；不得 amend 或回滚共享工作树中的其他改动。

---

## 最终验收清单

- [ ] direct_test_e0 严格使用 test 的 energy - atomization_energy，并在 report/图中标记 test-informed/transductive。
- [ ] model_aware_val_e0 逐 member 使用 val total energy 拟合 minimum-norm Delta_E0，test 不参与拟合。
- [ ] 方法二在 val forward 前按无标签结构身份排除 2 个 test overlap；test 保持不变，公开 manifest 只记录规则和数量。
- [ ] test model forward 次数为零；val forward 明确 compute_force=False、compute_stress=False。
- [ ] corrected prediction 通过原 fge.prediction.v1 validator，且只改变 energy_members 和重建的 references。
- [ ] 两方法 force members、mapping、weights、raw prediction 和 checkpoint 均未改变。
- [ ] published manifest 不绑定旧来源；internal integrity audit 不发布、不回传。
- [ ] hard contract 抛 error，质量问题只 warning；W&B 只记录有限标量。
- [ ] 两方法各 43 PNG + 43 PDF，无 Stress，标题和 audit 含正确方法标签。
- [ ] 本地仅存在可发布代码与 ignored 图片；没有迁移大数据、prediction、日志或 W&B artifact。
- [ ] 当前分支提交只包含 FGE 范围及必要的根 .gitignore 变更。
