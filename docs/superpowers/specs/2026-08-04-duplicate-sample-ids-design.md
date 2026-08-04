# ConfidenceHead 重复样本唯一 ID 设计

## 背景与根因

MATPES production 训练集包含 348,780 条结构，其中存在两组内容完全相同的样本：

- 索引 175647 与 223139；
- 索引 87271 与 307312。

当前 `structure_id(atoms)` 对原子序数、坐标、晶胞、周期边界、参考能量和参考力生成 SHA-256 内容哈希。相同内容必然得到相同 ID。现有 production 数据校验只检查不同 split 之间的内容重叠，没有检查单个 split 内重复；而 CacheWriter 要求缓存中的 `structure_id` 唯一。因此任务 11430 在 train 索引 223139 遇到第一条重复样本时，抛出 `CacheCorruptionError`。

这不是 GPU、并发、断点续建或 shard 损坏。任务失败时 `progress.pt` 已正确提交到 `next_index=215286`，17 个 shard 均有效。该 4.2 GB 未完成缓存已在用户明确授权后删除。

## 目标

- 保留原始 train/validation/test 文件及其 SHA-256；
- 保留全部 348,780 条训练样本，包括两条重复出现；
- 同一 split 内内容相同的样本可以分别进入缓存、训练、预测和发布结果；
- 不放宽 production split 隔离：相同内容出现在 train、validation、test 的不同 split 中仍必须拒绝；
- sample ID 必须确定、唯一、可追溯到原内容哈希；
- 不改变 checkpoint、特征计算、标签、损失或训练权重。

## 双层身份模型

### Content ID

`content_id` 保持现有 64 位 SHA-256 语义，输入字段不变：

- 原子序数；
- 原子坐标；
- 晶胞；
- 周期边界；
- 参考能量；
- 参考力。

Content ID 只表达“两个样本的科学内容是否完全相同”，用于输入一致性核验和 production split 泄漏检查。

### Sample ID

数据集按文件顺序扫描，并为每个 content ID 单独维护从 0 开始的 occurrence 计数。sample ID 格式固定为：

```text
<64位 content_id>#<非负十进制 occurrence>
```

例如同一内容第一次和第二次出现分别为：

```text
9eb08c...b97758#0
9eb08c...b97758#1
```

所有样本都带 occurrence 后缀，包括只出现一次的样本，因此格式一致且不存在“有时是哈希、有时是复合 ID”的歧义。文件顺序和数据文件 SHA 已属于缓存身份；只要输入文件不变，sample ID 就稳定不变。

## 数据结构与接口

`DatasetHandle` 同时保存：

- `content_ids: tuple[str, ...]`：按数据集顺序排列的内容哈希；
- `structure_ids: tuple[str, ...]`：为兼容现有缓存和发布接口而保留的字段名，其值改为唯一 sample ID；
- 原有 `path`、`sha256`、`size`。

`load_dataset` 在一次顺序扫描中计算 content ID 和 occurrence-qualified sample ID，不删除、不排序、不合并样本。

`build_structure_batch` 增加可选的预期 sample ID 输入。ConfidenceHead cache workflow 必须传入当前切片的 sample IDs。构建图时仍从实际 Atoms 重新计算 content ID，并验证：

- sample ID 数量与结构数量一致；
- sample ID 格式合法；
- sample ID 的 content ID 前缀与实际结构重新计算的 content ID 完全一致。

验证通过后，batch、cache shard、训练记录、预测文件和发布结果沿用现有 `structure_id` 字段承载 sample ID。这样不需要改动所有下游 schema 字段名，同时每一行仍可从 `#` 前缀恢复 content ID。

不使用 ConfidenceHead dataset workflow 的通用 `build_structure_batch` 调用可以省略 sample IDs；这种情况下函数为每个输入结构生成 `<content_id>#0`。该兼容路径只适用于调用者已经保证输入中没有重复内容的独立 batch。

## Split 隔离

`validate_split_isolation` 必须比较 `content_ids`，而不是 sample IDs。否则相同内容在不同 split 中可能因为 occurrence 后缀不同而逃过泄漏检查。

规则为：

- 同一 split 内相同 content ID：允许；
- 不同 production split 之间相同 content ID：继续抛出 `DataContractError`；
- smoke-test profile 的现有复用行为保持不变。

## 缓存、身份与恢复

CacheWriter 仍要求 `structure_id` 唯一。由于传入的是 sample ID，同一 split 内重复内容不再触发缓存冲突；CacheWriter 本身无需放宽重复策略。

该代码修改会改变 git code identity，因此新的 cache ID 和缓存目录必然与失败任务不同。旧的未完成缓存已经删除，新任务必须从索引 0 重新构建。后续正常中断仍使用现有 `progress.pt` 与 `next_index` 机制续建；sample ID 由输入文件顺序确定，恢复后不会漂移。

九组 production 配置共享相同输入、特征 schema 和代码身份，因此仍共享一个新缓存目录。

## 错误处理

以下情况在写入缓存前立即拒绝：

- malformed sample ID；
- occurrence 不是非负整数；
- sample ID 数量与结构数量不一致；
- sample ID 的 content 前缀与实际结构内容不一致；
- production split 之间 content ID 重叠。

单个 split 内重复不再被视为损坏；真正重复的 sample ID 仍由 CacheWriter 拒绝，以捕获 resume、索引或调用方错误。

## 测试设计

采用 TDD，先建立以下失败测试：

1. 两个完全相同的结构在同一 DatasetHandle 中得到相同 content ID、不同 sample ID：`#0` 与 `#1`；
2. 不重复结构也统一得到 `#0`；
3. cache workflow 能完整写入同一 split 内的重复内容，最终结构数量保持不变；
4. build batch 保留调用方传入的 sample ID；
5. sample ID 与实际内容不匹配时在 MACE 推理前失败；
6. 相同 content ID 出现在不同 production split 时仍被拒绝；
7. resume 从 `next_index` 继续时，occurrence-qualified IDs 与首次扫描一致；
8. 原有缓存、训练、预测、发布和 n20 smoke tests 全部通过。

实现后运行完整 ConfidenceHead 测试套件，并在服务器重新加载九份 production 配置、核验输入 SHA 和 shell 语法。不会自动重新提交 Slurm 任务；重新提交属于独立的服务器状态变更，需要用户明确授权。

## 部署

- 在当前 `ConfidenceHead` 分支实施并提交；
- 推送 GitHub `ConfidenceHead` 分支；
- 服务器 `/home/bywang/code/UQ/mace_new` 仅执行 fast-forward 部署；
- 保留服务器已有的 production YAML 修改和未跟踪 `run/submit.sh`；
- 部署后不恢复或迁移已删除的旧缓存。
