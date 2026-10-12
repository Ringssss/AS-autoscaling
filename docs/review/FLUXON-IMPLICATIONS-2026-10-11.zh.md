# Fluxon 对 FluidKV 的启示

> 历史/提案留档：以[技术总览](../FLUIDKV-TECHNICAL-ARCHIVE.zh.md)标明的证据状态和最新结论为准；未实施的方案不代表已有能力。

核对对象：Tele-AI/Fluxon，固定 commit `cc800babb844092362d10369eacb6aa0685f1fde`。本次读取并核对 Git blob 身份的文件为 master_kv_router 下的 get.rs、msg_pack.rs、mod.rs、placement.rs；未部署 Fluxon，也未作全仓性能审计。

## 已核实的机制

- 当前读协议名为 GetStart/GetDone/GetRevoke，和用户博客的 GetPlan/Bind 用词不同。GetStart 会选源、分配或复用接收内存、尝试预留持久副本容量、持有源 allocation，并登记 inflight get。因此它不是无副作用的路由探针。
- GetStartResp 提供源节点、Memory/SSD 类型、地址和长度。Memory 路径在有效候选中随机选择；目标已有本地副本时重用目标副本。这个局部观察不能扩写成“Fluxon 全部调度都随机”。
- GetMetaResp 只有 exists、len 与错误字段，不能直接提供候选 serving worker 的完整恢复路径、链路积压或 GPU 可续算边界。
- InflightGetInfo 有源/请求者 generation、容量 reservation、terminal_claimed。GetDone/GetRevoke 与成员生命周期处理终态。不能描述成“Fluxon 完全没有在途记账或回滚”。
- placement.rs 的默认 PUT policy 为 RandomPlacementPolicy，同时提供 LocalFirstPlacementPolicy。它控制数据副本放置，语义不同于 LLM 请求落点。

## 对我们的设计含义

1. 路由评估和资源承诺应分开。只读候选查询不能替每个备选目标创建真实传输与容量预留；最终选择后再由数据面创建真实动作。
2. 控制器影子负载与数据面真实 reservation 是不同职责。把已提交动作按 ID 记入预计网络/GPU 工作，用真实完成校正；避免双重计费与两套资源所有权。
3. 数据对象可读和模型可续算之间仍有接口边界。FluidKV 的目录应提供经过运行时验证的 KV/recurrent 执行边界，不能仅从 GetMeta.exists 推断可执行命中。
4. 一个候选的恢复时间取决于其实际资源路径。GPU-direct 是可选数据路径，其价值需要与可行 CPU/DRAM 路径实测比较；CPU 参与调度不能证明 CPU 内存路径不可用。
5. Fluxon 提供数据面机制基础，不能据此声称 FluidKV 已有多级缓存端到端收益。本轮测量继续使用已验证的直接状态复制路径。

## 稳定源码链接

- [GetStart/GetDone/GetRevoke 与 GetMeta](https://github.com/Tele-AI/Fluxon/blob/cc800babb844092362d10369eacb6aa0685f1fde/fluxon_rs/fluxon_kv/src/master_kv_router/get.rs)
- [消息字段](https://github.com/Tele-AI/Fluxon/blob/cc800babb844092362d10369eacb6aa0685f1fde/fluxon_rs/fluxon_kv/src/master_kv_router/msg_pack.rs)
- [在途状态与终态所有权](https://github.com/Tele-AI/Fluxon/blob/cc800babb844092362d10369eacb6aa0685f1fde/fluxon_rs/fluxon_kv/src/master_kv_router/mod.rs)
- [PUT 放置策略](https://github.com/Tele-AI/Fluxon/blob/cc800babb844092362d10369eacb6aa0685f1fde/fluxon_rs/fluxon_kv/src/master_kv_router/placement.rs)
