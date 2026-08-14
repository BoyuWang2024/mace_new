# 内部旧结果迁移工具

此目录只用于认证和适配旧 MACE BootStrapping 结果，不属于公开发布包。

- 禁止调用训练、预测、聚合、UQ 或 analysis 计算入口。
- 模型与 latest resume 按字节复制并核对 SHA-256。
- tensor 只允许 CPU 加载、成员轴切片、字段重命名和 NumPy 序列化。
- 正式 run 中不记录旧服务器绝对路径；外部 audit 必须位于正式目标之外。
- 重复迁移必须先完成全量等价验证，然后返回 `written=False`。
