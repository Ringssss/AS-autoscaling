#!/usr/bin/env python3
"""Report observed structure separately from hypothetical placement candidates."""
import argparse
import json
from pathlib import Path

import numpy as np


def main(args):
    root = args.root
    agent = json.loads((root / "agentx/analysis.json").read_text())
    details = json.loads((root / "agentx-native-details/analysis.json").read_text())
    kimi = json.loads((root / "kimi/analysis.json").read_text())
    c, n = agent["native_counts"], agent["native_counts"]["requests"]
    assert c["requests"] == details["native_counts"]["requests"]
    for key in ("completed_reuse_32768", "completed_reuse_and_other_active_32768", "context_fit_requests"):
        assert c[key] == details["native_counts"][key]
    assert agent["sha256"] == details["sha256"]
    sensitivity = []
    for workers in (2, 4, 8):
        group = [v for k, v in agent["placement_sensitivity"].items()
                 if k.startswith(f"n{workers}-") and k.endswith("ttl300")]
        all_fractions = [v.get("load_mismatch_missing_state_32768", 0) / v["requests"] for v in group]
        fit_fractions = [v.get("fit_load_mismatch_32768", 0) / v["context_fit_requests"] for v in group]
        sensitivity.append({"workers": workers, "ttl_seconds": 300, "seeds": 3,
                            "fraction_all_min_max": [min(all_fractions), max(all_fractions)],
                            "fraction_fit_min_max": [min(fit_fractions), max(fit_fractions)]})
    tables = {"agentx": c, "agentx_gap": agent["finish_to_next_arrival_gap"],
              "agentx_peak": agent["per_conversation_peak_logged_requests"],
              "placement": sensitivity,
              "kimi": {k: kimi[k] for k in ("requests", "identity_fields", "input_tokens", "output_tokens", "ttft")}}
    (root / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    lines = ["# Native trace 中的状态准备空间", "", "2026-10-10；本轮只做 CPU 上的原始 trace 核验和离线结构分析。", "",
             "## 结论", "",
             "AgentX/Weka 的长前缀跨轮复用很普遍，并与一定数量的同模型并行活动重叠。"
             "这支持为 routing 增加选择性的状态准备能力。实际 worker 放置、设备负载和全局到达未知，"
             "所以 trace 不能直接给出有净收益迁移的发生频率。Kimi 的匿名日志更适合长度/延迟/缓存分布，不能恢复这个因果问题。", "",
             "## 数据发现和完整性", "",
             "远端 main / HEAD：`2e96af93f091808ed9e31f04eed800d0285c97b4`，与本地 HEAD 一致。"
             "Kimi 数据位于 GitHub Release，而不是 Git 主分支的大文件："
             "[kimi-trace-2026-04-07](https://github.com/Ringssss/AS-autoscaling/releases/tag/kimi-trace-2026-04-07)。", "",
             "本地 Kimi archive：`experiments/multinode/data/kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07.tar.gz`，"
             "58,121,430 bytes；SHA256 `e7d8fd50757208220b9bc5bbd111a9d491b94a89a6c42390ab345aab9a0c3513`，与 Release API digest 一致。"
             "重新解析24个小时文件，共245,555条，并核对归一化记录、原文件hash和历史审计。", "",
             f"AgentX/Weka 原文件：`{agent['source']}`，SHA256 `{agent['sha256']}`；"
             "重新扫描393条conversation，共98,827个模型请求、2,535个stream/model序列。"
             "原始模型标签分开；hash只在同conversation内解释；前序input只在完成后作为已知状态。", "",
             "## AgentX 的原生证据", "",
             "| 属性 | 数量 | 占全部请求 |", "|---|---:|---:|"]
    fields = [("已完成同stream输入可复用≥8K", "completed_reuse_8192"),
              ("已完成同stream输入可复用≥32K", "completed_reuse_32768"),
              ("已完成同stream输入可复用≥64K", "completed_reuse_65536"),
              ("复用≥32K且同模型其他stream正在执行", "completed_reuse_and_other_active_32768"),
              ("与活动中的其他stream共享≥32K输入前缀", None)]
    for label, key in fields:
        value = c[key] if key else details["native_counts"]["arrivals_sharing_32768_with_other_stream"]
        lines.append(f"| {label} | {value:,} | {value/n:.2%} |")
    fit = c["context_fit_requests"]
    lines += ["", f"原始input+output≤131,072且output>0的请求有{fit:,}条（{fit/n:.2%}）。"
              f"这个可运行子集中，≥32K已完成前缀复用有{c['fit_completed_reuse_32768']:,}条"
              f"（{c['fit_completed_reuse_32768']/fit:.2%}）；同时有其他同模型stream活动的有"
              f"{c['fit_reuse_and_other_active_32768']:,}条（{c['fit_reuse_and_other_active_32768']/fit:.2%}）。", "",
              f"完成到后续到达的时间差p50={agent['finish_to_next_arrival_gap']['p50']:.3f}s，"
              f"p90={agent['finish_to_next_arrival_gap']['p90']:.3f}s。这不是经验证的工具执行时长。"
              "旧rawcensus中的11.879s是到达间隔，不能当成迁移窗口。", "",
              f"连续≥32K复用序列的长度p50={agent['consecutive_long_reuse_runs']['p50']:.0f}、"
              f"p90={agent['consecutive_long_reuse_runs']['p90']:.0f}；可以支持多轮复用的动机，"
              "不能直接推导一次副本可用多久，因为副本驻留和新增状态未知。", "",
              f"每conversation观察到的API重叠峰值：p50={agent['per_conversation_peak_logged_requests']['p50']:.0f}，"
              f"p90={agent['per_conversation_peak_logged_requests']['p90']:.0f}，max={agent['per_conversation_peak_logged_requests']['max']:.0f}。"
              f"同模型且共享≥32K前缀的同时活动distinct streams最大为"
              f"{details['native_counts']['max_simultaneous_distinct_streams_same_prefix_32768']}。"
              "因此10月9日64分支同长前缀的构造不能当成这份corpus的典型fan-out。", "",
              "## 放置敏感性：联合条件的候选频率", "",
              "逐conversation保持原始相对时间，每个模型设兼容worker池，stream固定哈希到worker。"
              "只在原记录完成时登记input状态，按300s last-use TTL观察；不执行迁移、不改变到达。"
              "候选条件：source已有≥32K前缀、source的观察活动数比最少活动target高、target缺少source的一部分状态。", "",
              "| worker数 | 3种hash seed：占全量请求 | 占131K可运行子集 |",
              "|---|---:|---:|"]
    for s in sensitivity:
        a, b = s["fraction_all_min_max"], s["fraction_fit_min_max"]
        lines.append(f"| {s['workers']} | {a[0]:.2%}–{a[1]:.2%} | {b[0]:.2%}–{b[1]:.2%} |")
    lines += ["", "这些是声明放置假设下的结构候选，不是原集群的热点率、排队率或收益覆盖率。"
              "正在服务1个请求的worker仍可能立即接收新请求并batch，所以活动数差不等于可节省等待。"
              "不同conversation的时间均从零开始，不将它们直接叠加为真实全局负载。"
              "当前cache仅有TTL，没有字节容量、锁定、驱逐、历史output状态或真正Mamba快照边界。", "",
              "## 准备窗口与状态字节的敏感性", "",
              "在≥32K已完成同stream复用的请求上，用0.02s固定开销、额外64MiB状态及声明的bytes/token、"
              "有效带宽估计复制时长，与finish-to-arrival间隔比较。这是计算条件，不是实测transport。", "",
              "| 状态表示 | 有效带宽 | 估计复制能放入间隔的比例 |", "|---|---:|---:|"]
    for bpt in (20480, 131072):
        for bw in (5, 25):
            hits = details["long_reuse_gap_counts"][f"reuse32768_copyfits_{bpt}B_{bw}GBs"]
            lines.append(f"| {bpt/1024:.0f}KiB/token + 64MiB | {bw}GB/s | {hits/c['completed_reuse_32768']:.2%} |")
    lines += ["", "准备时长会随表示和网络明显变化；工具期发生重算或数据移动，还需扣除资源争用。"
              "有窗口只表示可以准备，不表示应该迁移。", "",
              "## Kimi：原生长度与缓存证据", "",
              "所有245,555条日志的input/output均脱敏，conversation_id/routing_key/extra_key/session_params均无非空值。"
              "没有精确前缀身份、Agent lineage、工具边界或足以恢复所有权的worker事件。", "",
              "| 属性 | 数量 | 占全部请求 |", "|---|---:|---:|"]
    for cohort in kimi["cohorts"]:
        if cohort["threshold"] in (32768, 65536):
            label = "input" if cohort["kind"] == "long_input" else "历史路径记录的cached tokens"
            lines.append(f"| {label}≥{cohort['threshold']} | {cohort['requests']:,} | {cohort['fraction_all']:.2%} |")
    lines += ["", f"TTFT p50={kimi['ttft']['p50']:.3f}s、p90={kimi['ttft']['p90']:.3f}s，"
              f"output p50={kimi['output_tokens']['p50']:.0f}、p90={kimi['output_tokens']['p90']:.0f} tokens。"
              f"{kimi['cache_invalid_rows']:,}条cache count不合法，统计cached时排除。", "",
              "Kimi的≥32K input比例不能作为可搬运长历史的比例，cached count较小也不能证明其他worker没有状态。"
              "无法从匿名请求恢复源、目标与下一轮，因此不能用该trace宣称迁移频繁获益或无效。"
              "历史trace-shaped replay使用其他数据内容和人为lineage，其收益不能作为原生Kimi的机会频率。", "",
              "## 已有模拟的复核", "",
              "旧`experiments/weka_baselines/oracle/frontier.py`的build_events虽写保持intra-session timing，"
              "实际乘scale压缩了每条stream的内部时间；simulate还在服务完成前admit缓存，且存在用历史busy累计量选目标的策略。"
              "本报告不使用该frontier结果作为全局服务或迁移机会证据。", "",
              "## 适合的系统定位", "",
              "将本系统定位为routing可调用的状态准备层：router继续选择执行位置，准备层为值得改善的目标"
              "提供经过确认的可执行副本与准备成本。已有热目标、原地batch可满足目标或收益不足时直接保留原routing；"
              "准备失败/过期时回到原可行路径。fallback会保留已发生的资源开销，不能承诺零回归。", "",
              "主比较采用同一个router的原始能力与加准备层之后的能力；强router已有远端取回、预取或复制时保留这些能力。"
              "机制收益由确认的新增选择、避免重算与完整任务/集群代价归因。"
              "已存在DualMapAdapter和SnapshotDirectory可作为集成入口；目前未将本次观察升级为完整已实现能力。", "",
              "原生证据说明值得做选择性集成，不能支持把84.6%复用率或5%–9%结构候选直接乘17s作为预计系统收益。", ""]
    (root / "REPORT.zh.md").write_text("\n".join(lines) + "\n")
    plot(root, agent, details, kimi, sensitivity)
    print("\n".join(lines))


def plot(root, agent, details, kimi, sensitivity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    c = agent["native_counts"]
    values = [c["completed_reuse_32768"], c["completed_reuse_and_other_active_32768"],
              details["native_counts"]["arrivals_sharing_32768_with_other_stream"]]
    axes[0].bar(["Completed reuse", "+ other active", "Cross-stream sharing"],
                [100*v/c["requests"] for v in values], color=["#355c7d", "#6c8caa", "#93afc6"])
    axes[0].set(title="AgentX: observed >=32K structure", ylabel="% of all requests")
    low = np.array([s["fraction_all_min_max"][0]*100 for s in sensitivity])
    high = np.array([s["fraction_all_min_max"][1]*100 for s in sensitivity])
    axes[1].bar([str(s["workers"]) for s in sensitivity], (low+high)/2,
                yerr=(high-low)/2, capsize=4, color="#c57b35")
    axes[1].set(title="Hypothetical stationary placement\nStructural candidates, not speedup", xlabel="Compatible workers / model", ylabel="% of all requests")
    cohorts = [next(x for x in kimi["cohorts"] if x["threshold"] == 32768 and x["kind"] == k)
               for k in ("long_input", "observed_long_cache")]
    axes[2].bar(["Long input", "Long observed cache"], [100*x["fraction_all"] for x in cohorts], color="#507b63")
    axes[2].set(title="Kimi: observed >=32K counts\nSession/prefix identity unavailable", ylabel="% of all requests")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(root / "trace-space.png", dpi=180)
    fig.savefig(root / "trace-space.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/native-relocation-space-20261010"))
    main(p.parse_args())
