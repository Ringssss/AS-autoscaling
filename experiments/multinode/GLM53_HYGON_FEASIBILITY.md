> **更正（FP8硬件核验）：本地BW实测为gfx936，`is_hcu_native_fp8_supported()`返回False；不能按原生FP8计算路径直接部署。下文16卡容量结论只适用于保留低比特常驻、且存在可用计算后端的前提。默认hipBLASLt小矩阵探测已报告无合适算法并产生非有限输出，不能用作有效推理。需验证高精度计算的低比特权重路径，或转换到受支持的INT8/INT4格式；全量BF16展开会使16卡容量不足。

# GLM-5.3：本地海光 DCU / SGLang 部署可行性

检查日期：2026-09-23。仅进行了公开模型元数据、源代码、容器版本、设备占用及文件系统的只读检查；未下载权重、未启动推理、未停止旧容器。

结论：**FP8 单实例在16张64GiB DCU上具有合理容量基础，本地海光版SGLang也有相关模型与算子路径，值得做兼容性验证，但尚未实测证明可直接启动。现有六卡配置不够；三节点不等于三份可独立运行的副本。磁盘空间在数据盘上足够，/home不足。**

## 权重与模型

官方模型 `zai-org/GLM-5.3`，检查时版本 `aca966e4e02791568aa6a4ced368624b3d897f42`。

- 架构：`GlmMoeDsaForCausalLM`，78层、hidden6144、64注意力头、256路由专家、每token激活8个专家、MLA/DSA、1个MTP层。
- 配置中的`dtype=bfloat16`不表示全BF16权重；`quantization_config`明确为动态FP8 e4m3、128×128 block量化。
- 官方FP8共141个safetensors分片；文件合计755,632,050,320 bytes，约755.6GB / 703.7GiB。索引tensor总大小755,617,140,416 bytes。
- HF统计的权重张量元素总数约753.3B，MoE不能只按每token激活参数估算常驻模型内存。
- 模型标注上下文上限1,048,576，不代表本硬件可在该长度和任意并发下运行。
- 官方还提供GLM-5.3-BF16，SGLang说明其权重约1.5TB。不是本次首选。

## DCU 与分布式布局

三个节点均为8×海光BW，每卡65,520MiB（约64GiB）。下表只是FP8权重理想平均分摊，未计复制、对齐、KV、工作区、通信、运行时及加载峰值。

| 总卡数 | 总显存GiB | 理想权重GiB/卡 | 判断 |
|---|---:|---:|---|
| 6 | 383.9 | 117.3 | 全DCU常驻不可行 |
| 8 | 511.9 | 88.0 | 全DCU常驻不可行 |
| 12 | 767.8 | 58.6 | 极紧，不建议作为可靠服务配置 |
| 16 | 1023.8 | 44.0 | 有合理余量，优先验证 |
| 24 | 1535.6 | 29.3 | FP8容量更宽裕，但目前不能独占24卡 |

即时占用：

- node27：无空卡；0–1为本用户`zhujianian-qwen36-check`，2–3为`zhujianian-agentshift-hygon-source`，4–5为`zhujianian-agentshift-hygon-destination`，6–7为`zhujianian-as-multinode-pilot`的设备配置。每卡现占约36–56GiB。释放这些旧任务后有望恢复8卡；实际释放后还需核验无其他进程。
- node28：2、6、7号卡接近空闲，4–5为本用户pilot；0–1各占约53GiB，3号卡约19GiB，不当作空闲卡。释放本用户pilot后也只有约5张完整可用卡。
- node29：2–7号卡接近空闲，0–1为本用户pilot。释放本用户pilot后有望恢复8卡。

优先候选：**node27+node29，TP16、2节点、单实例**。这样64注意力头与256专家均可按16整分，MoE中间维2048按16切分后为128，与block量化粒度相容；这些是静态必要条件，不是运行成功证明。

若将三个节点全腾空，TP8×PP3是另一候选；不要直接写TP24，64个注意力头不能按普通TP24均分。现有代码有PP和跨阶段top-k传递处理，但完整GLM-5.3/DCU路径仍需验证。当前node28的占用不支持完整TP8阶段。

不建议三节点各4卡仅为用满节点数而做TP4×PP3：每卡权重约58.6GiB，剩余空间过小。BF16在24卡上同样接近容量边界，不作为优先方案。CPU offload是另一种性能／实现取舍，本次没有评估为可用方案。

## SGLang / 海光支持程度

当前实验容器实际版本：

- SGLang `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025`
- PyTorch `2.10.0+das.opt1.dtk2604.2608131652.g28498c`
- Transformers `5.6.0`

已核验容器内模型代码摘要与本地提取源码相符。

本地存在`GlmMoeDsaForCausalLM`，NSA/DSA后端有`is_hcu()`分支，FP8和MoE也有海光路径。GLM-5.3配置的`index_topk_freq=4`、`index_skip_topk_offset=3`与本地索引共享规则一致，计算出的full层序列与配置`indexer_types`相符；权重索引中的indexer命名亦与本地实现的命名族相符。这比仅看到“支持GLM模型”更有依据。

官方模型卡说GLM-5.3与5.2使用同一base model，SGLang cookbook明确支持5.3。但cookbook展示的是NVIDIA和AMD MI300X/MI325X/MI355X，不能当成海光已经验证。配置由Transformers5.15.0生成，本地为5.6.0，也需核对配置／tokenizer语义，不能仅据版本号断言兼容或不兼容。

需要实际验证的点：FP8 block量化内核、DSA prefill/decode和indexer、跨节点通信、权重加载峰值、输出正确性。不能照抄NVIDIA FlashMLA/DeepGEMM/NVFP4参数；海光并不自动等同AMD ROCm。当前本地standalone MegaMoE路径还明确拒绝blockwise FP8，不能随意启用它。

首次部署应选择短上下文、低并发、关闭MTP及复杂重叠优化，确认基本前向与输出后再扩大。此处为方案建议，本次没有执行。

## 磁盘和主机内存

| 路径 | 空闲（df -h） | 挂载与可用性 |
|---|---:|---|
| /home | 156GiB | node27本地盘；node28/29通过NFS共享同一份，不能相加 |
| /data/node28 | 约1.6TiB | 独立本地数据盘，其他节点有NFS挂载；`shared`可写 |
| /data/node29 | 约1.4TiB | 独立本地数据盘，其他节点有NFS挂载；本用户目录可写 |
| node27 / | 约4.3GiB | 根盘紧张 |
| node28 / | 约470MiB | 根盘几乎满 |
| node29 / | 约20GiB | 根盘余量有限 |

FP8建议至少预留0.9–1.0TiB工作空间（约704GiB权重之外还需下载临时文件等）。**可放`/data/node29/zhujianian/GLM-5.3`，或在`/data/node28/shared`下建立本用户子目录。** 一份共享权重即可被多节点访问，不需要在同一NFS挂载上重复下载三份。共享盘加载带宽和多进程页缓存压力仍需考虑。

不要把下载缓存或新镜像默认放在已经接近满的根盘。若缓存和最终目录各保留一份实体副本，会接近两倍权重占用；应明确下载方式，避免无意复制。远端大容量共享文件系统虽显示大量空闲，但未核验个人配额，本结论不依赖它。

三个节点RAM均约503GiB，可用分别约385/435/457GiB。常规分片流式加载具有容量基础；每进程完整复制权重的加载方式不合适。

## 最终建议

优先FP8，用数据盘存一份权重，释放本用户旧容器后以node27+node29的16卡验证一个实例。三节点24卡FP8理论更宽裕，但当前node28资源不满足；以前的三节点六卡配置明确不够。

结论置信度分开看：权重大小和磁盘余量已确认；16卡内存可行性是预算判断；海光SGLang能否正确完成推理尚未实测，不能承诺直接启动成功或吞吐性能。

来源：

- https://huggingface.co/zai-org/GLM-5.3
- https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json
- https://huggingface.co/zai-org/GLM-5.3/blob/main/model.safetensors.index.json
- https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3
- 模型元数据通过hf-mirror和ModelScope获取，未下载模型权重。

## 后续资源释放与写权限验证（2026-09-23）

按用户要求，已停止此前六个模型／实验容器：node27的pilot、hygon-source、hygon-destination、qwen36-check，以及node28/29各一个pilot。三个节点均无运行中的`zhujianian`容器。其他用户服务未停止。

停止后DCU显存复核：node27八卡各约5MiB，node29八卡各约2MiB；node28的2/4/5/6/7号卡各约4MiB，其余三卡仍有其他任务。当前可用完整卡为8+5+8，共21卡，node27+node29的16卡已实际腾空。

`/data/node29/zhujianian`、新建的`/data/node28/shared/zhujianian`均完成临时文件写入、fsync、读回一致性验证，测试文件已删除。三个节点对这两个用户目录均有写权限。数据盘可以实际存放本用户文件；此检查未下载模型。
