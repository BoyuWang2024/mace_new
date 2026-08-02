# ConfidenceHead 训练与发布指南

## 1. 范围与数据流

本模块把冻结的 MACE 势函数输出转换为可训练的分类式置信度头。完整数据流是：严格加载 YAML 并验证输入 SHA-256；对三个 split 读取结构；在 cache build 阶段运行一次冻结 MACE，同时取得预测、参考标签和 `products.0/1` 特征；把连续张量按完整结构写入 cache-v2 shard；只使用 train shard 拟合固定分箱；仅从 cache 训练 Force 与 Energy head；最后重建身份、日志和 checkpoint 关系并完成原子化终结。

`build_cache` 是唯一加载 MACE 的阶段。`fit_bins`、`train` 与 `check_training` 都不加载 MACE，也不重新计算力或能量。这样既避免重复昂贵计算，也让训练输入成为可校验、可恢复、与 backbone 版本绑定的不可变证据。生产 split 必须使用不同文件，并在结构级检查重叠；只有 `smoke_test` profile 允许复用同一小数据文件。

## 2. 连续误差与模型数学

对原子 i 的 Force 分量 c，连续绝对误差为 `a_i,c = abs(F_pred_i,c - F_ref_i,c)`。`atom_mean` 模式定义 `e_force_i = mean_c(a_i,c)`，每个原子产生一个标签，logits 形状为 `[N_atom, B_force]`。`component` 模式保留三个标签，形状为 `[N_atom, 3]`；x、y、z 三个 head 拥有相同网络结构但参数彼此独立，交叉熵前把它们视为 `3*N_atom` 个真实样本。

结构 s 的 Energy 标签采用每原子绝对误差：`e_energy_s = abs(E_pred_s - E_ref_s) / N_atom_s`。因此大结构不会仅因原子数更多而隐式放大 Energy loss。每个结构先对 640 维逐原子特征计算 raw moments，再递归计算 cumulant：`kappa_r = m'_r - sum C(r-1,j-1) * kappa_j * m'_(r-j)`。一阶是均值；二阶及以上使用 signed root，即 `sign(kappa_r) * abs(kappa_r)^(1/r)`，避免奇偶阶符号信息丢失和量纲快速膨胀。

从一阶到 K 阶的统计量拼接后进入可训练的 `Linear(640×K, 512)`、LayerNorm、Dropout 和 Energy head。投影层属于模型参数，必须参与优化、checkpoint 保存、严格参数 schema 比较和恢复。每个隐藏块固定为 Linear、SiLU、LayerNorm、Dropout，末层输出 bin logits。

硬标签损失为 `L_total = w_force * mean(CE_force) + w_energy * mean(CE_energy)`。先在每个启用分支按真实样本数求交叉熵均值，再加权；epoch 汇总也按样本数累计，不能平均 batch mean。权重为零的分支完全禁用，不创建 bins 或参数；两个权重同时为零是非法配置。第一阶段没有 soft label、label smoothing、class weight 或回归损失。

## 3. 固定与训练分位分箱

默认 `fixed_linear_v1` 给定最大误差 M 和 B 个 bins：`width=M/B`，阈值为 `threshold_i=i*width`，代表值为 `(i+0.5)*width`。区间统一采用左闭右开：误差精确等于阈值时进入右侧 bin。误差超过 M 仍放入最后一类并累计 `overflow_count`，不丢弃，不裁成新样本，也不添加无穷代表值。

`train_quantile_log_v1` 只读取 train errors。正误差的 5% 线性插值分位点作为 lower anchor，全量误差的 99.5% 线性插值分位点作为 upper anchor；B-1 个阈值在两个 anchor 的 log 空间等距生成。非空 bin 代表值是该 bin 训练误差的中位数，空 bin 使用版本固定的确定性解析 fallback。高于 upper anchor 的值进入最后一类并计为 overflow。没有正误差、anchor 非有限或 `lower >= upper` 都会失败，而不是猜测替代值。

阈值、代表值、计数、来源和 overflow 会写入不可变 `binning.pt` 及其 manifest。validation 和 test 绝不参与拟合。训练与验证只把连续误差按已经锁定的阈值变成 hard labels；恰好等于阈值的相等语义在所有阶段保持一致。

## 4. 四层身份链

`cache_id` 绑定 checkpoint 内容 SHA、数据内容 SHA、固定特征模块和维度、cache schema/formula、数据读取语义、代码身份。任何输入字节或相关代码改变都会形成不同缓存，旧目录不会被覆盖。

`binning_id` 绑定 `cache_id`、算法版本、Force target mode、启用分支、bin 数、fixed 最大误差或固定 log-quantile 规则，以及最终拟合出的 thresholds、representatives、counts、sources、overflow。它说明标签边界究竟是什么。

`experiment_id` 在拟合前由规范化配置、`cache_id` 与代码身份生成，用于确定稳定运行目录。batch size、optimizer、严格确定性开关等训练语义变化必须改变它；W&B 在线或离线状态不进入科学身份。

`run_id` 最终绑定 `experiment_id` 和 `binning_id`。checkpoint、事件、摘要、validation 和 completion manifest 必须引用同一组四层身份。目录名相似不是相同实验，文件名相同也不能替代内容身份比较。

Git clean commit 与 dirty 工作树属于不同代码身份。允许 dirty Git 运行，但会记录 `git_dirty: true`，并将规范化实际 diff 的 `git_diff_sha256` 纳入身份；因此 dirty 结果不能冒充同 commit 的 clean 发布结果。发布前应尽量使用已提交且干净的代码，并把 commit、dirty 状态和环境记录一并归档。

## 5. cache-v2、不可变分箱与遍历顺序

cache-v2 的每个 shard 只保存 tensor 和基础类型，包括结构 ID、原子序数、原子 offsets、640 维特征、Force/Energy prediction 与 reference。一个结构绝不跨 shard。feature dtype 在同一 cache 内统一，预测与参考保留各自来源 dtype；拟合连续误差时统一提升为 CPU float64。manifest 记录每个 shard 的 SHA、范围、结构数、原子数、shape 和 dtype。

构建进度只在一个完整 shard 原子提交后前移。恢复前重新核验已经提交的全部 shard；只有 complete manifest 存在且重新验证通过，后续阶段才能读取缓存。binning 也采取“一旦存在即严格重验”的不可变规则：artifact 和 manifest 必须成对出现，内容哈希与身份必须完全相符，半成品不会被默默覆盖。

epoch 采样以完整结构为单位。每个 epoch 使用 `seed + epoch` 派生确定性结构 shuffle，再按配置 batch size 拼接原子和 offsets。中断恢复会从上一个完整 epoch 使用相同 seed 重建相同顺序；不会保存或恢复一个未提交 epoch 的半批次。validation 不 shuffle，并在 `eval` 与 `no_grad` 下按真实样本数汇总。

## 6. 选择、早停与精确恢复

每个 epoch 完整计算 train 和 validation 的 Force、Energy、加权 total loss。只有 validation total loss 严格降低时才原子更新 `best.pt`；相等不算改进。早停 bad-epoch 计数与 best 使用同一个 total 定义，达到 patience 或 max epochs 后正常结束。

`best.pt` 只含发布推理所需模型状态、参数 schema、最佳 epoch、validation 指标和身份。它是后续正式评估的默认 checkpoint，但不是续训入口。`last.pt` 是 epoch 边界恢复状态，额外保存 next epoch、optimizer、early-stopping、best 元数据、Python/NumPy/Torch CPU/Torch CUDA RNG；恢复时参数名、shape、dtype、optimizer、RNG 和身份必须逐项相符。

事件记录完成、best 判断和完整 epoch 状态准备好后，才提交 `last.pt`。如果进程在 epoch 内中断，该 epoch 未提交的工作被放弃；下一次从旧 `last.pt` 重新运行。若目录身份相同、有合法 last 且 `resume: true`，可以续训；身份不符、已完成 run 再训练、残缺目录没有合法 last、事件顺序损坏，都必须封闭失败。

## 7. 本地日志与 W&B

`events.jsonl` 是权威记录。每行使用规范 JSON、立即 flush 并 fsync，epoch 必须严格递增；恢复时只容忍崩溃产生的最后一行截断，不容忍中间坏行、重复 epoch 或字段漂移。训练 summary 和 overflow 统计从这些可信状态与分箱清单汇总。

W&B 只镜像已经形成的本地事件。`auto` 以有限超时尝试 online，通信或认证不可用时转 offline；显式 `offline` 从一开始写入 run 的 `wandb/`。W&B 初始化和记录使用隔离 RNG 状态，不能扰动训练 shuffle、dropout 或恢复等价性。联网后对具体 `offline-run-*` 目录执行 `wandb sync`，同步失败也不改变本地产物身份或完成状态。

## 8. 用户可见产物树

一个 name prefix 下包含 `cache/<cache_id>/` 和 `runs/<run_tag>-<experiment_id>/`。cache 目录包含 `progress.pt`、各 split 的 `shard-*.pt` 与最终 `cache_manifest.json`。run 目录的 `config/` 保存 source YAML 与 resolved JSON；`identity/` 保存 run、environment 和 Git 记录；`binning/` 保存 artifact 与 manifest；`run/` 保存事件、best、last、summary、validation、completion manifest 和 W&B 镜像。

source YAML 用于人类复核，resolved JSON 用于显示路径解析和规范化语义；两者不能互相替代。`.pt` 文件按内部安全 schema 读取，只保存 tensor/基础类型，不序列化任意模型对象。manifest 记录关联身份和内容 SHA，可发现复制不全、位翻转、手工编辑或拿错 checkpoint。

`training_validation.json` 给出核验细节；`training_manifest.json` 是唯一完成标记。终结检查会验证 schema、所有文件 SHA、tensor 有限性、精确参数集、事件顺序、best epoch 是否真为最小 validation total、last 的完成状态、禁用分支无残留、四层身份闭合。只有这一步成功后才可把结果列入发布候选。

## 9. 封闭失败与现场诊断

下列情况立即非零退出：输入 SHA-256 不符；production split 相同或结构重叠；标签缺失；元素不受 checkpoint 支持；预测、参考或特征含 NaN/Inf；hook 缺失、重复或维度错误；shard 缺失、hash 损坏、offset 不连续；阈值非严格递增；身份冲突；checkpoint 参数缺失、多余、shape/dtype 错误；optimizer/RNG/早停状态不完整；严格确定性遇到不确定算子。

遇到 partial finalization 时先保留现场：不要删除目录，不要移动身份文件，不要手工修改 JSONL，不要覆盖 best/last。检查是否已有合法 `last.pt`、事件末行是否仅为截断、binning artifact/manifest 是否成对、validation 是否已写但 completion manifest 尚未提交。保留现场后复制到独立位置分析；修复代码或输入时应产生新的代码身份，而不是伪装成原 run。

如果已有 `training_validation.json` 而没有 `training_manifest.json`，运行仍未完成。重新执行 check 只会在全部输入仍一致时安全终结；发现差异会失败并保留证据。训练和检查阶段若出现 MACE 加载通常说明流程边界被破坏，应停止并定位调用，而非增加 checkpoint 路径 workaround。

## 10. 生产预检清单

正式全量计算前按以下预检清单逐项确认：

1. 使用目标 commit，记录 `git status`，确认是否接受 dirty 身份。
2. 激活经过验证的环境，确认 PyTorch、MACE、ASE、W&B 版本和 CUDA 可用性。
3. 对 checkpoint、train、validation、test 真实文件重新计算 SHA-256，与 YAML 完全一致。
4. 确认 production 三个 split 文件不同，并完成结构级隔离检查；禁止复用 n20 语义。
5. 审阅 Force target 是 `atom_mean` 还是 `component`，确认两个 loss 权重和启用分支。
6. 审阅 fixed/log 算法、bin 数、最大误差、Energy cumulant order、投影与 head 宽度。
7. 确认 GPU 显存、cache/output 磁盘空间、文件权限和预期运行时长。
8. 确认 deterministic、seed、batch size、max epochs、patience、optimizer 都是本实验规范。
9. 确认 W&B 选择 auto/offline，且即使服务不可用也能保存本地权威日志。
10. 先以 n20 完成一次管线测试，但明确隔离其输出，绝不把 smoke 指标写入科学报告。

## 11. 运行后发布清单

计算完成后按发布清单检查：

1. 四个脚本均返回零，且最终 `training_manifest.json` 存在。
2. `training_validation.json` 的 valid 为真，所列 SHA 与实际文件一致。
3. cache、binning、experiment、run 四个 ID 在所有 artifact 中闭合一致。
4. best epoch 与事件中最小 validation total 一致，last 显示正常完成。
5. Force/Energy 启用状态与模型参数、bins 和指标字段完全一致。
6. 检查 Force 与 Energy 的 overflow_count；异常比例必须解释，不能隐去。
7. 确认事件没有重复/缺失 epoch，summary 可由权威 JSONL 重建。
8. 确认 `best.pt` 用于发布评估，`last.pt` 仅作为恢复审计证据。
9. 归档 source/resolved config、Git/environment 身份、binning manifest 和验证报告。
10. 若 W&B 为 offline，可在联网后同步，但以本地 manifest 为发布判据。
11. 不把 n20 的复用 split 指标当作科学结果；正式结论只来自隔离的 production 数据。
12. 在进入测试集评估、绘图或公开 bundle 前使用独立阶段规范，不在训练目录内手工追加未校验结果。
