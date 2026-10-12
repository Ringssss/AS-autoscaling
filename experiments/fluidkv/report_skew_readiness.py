#!/usr/bin/env python3
"""Recompute case tables from all retained p06 requests, without changing data."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def analyze(directory):
    data = json.loads((directory / "summary.json").read_text())
    requests = [json.loads(line) for line in (directory / "requests.jsonl").open()]
    by_trial = defaultdict(list)
    for row in requests:
        by_trial[row["repetition"], row["policy"]].append(row)
    rows = []
    for trial in data["trials"]:
        rep, policy = trial["repetition"], trial["policy"]
        records = by_trial[rep, policy]
        tasks = defaultdict(list)
        for record in records:
            tasks[record["consumer"]].append(record)
        task_seconds = []
        task_good = 0
        for records_for_task in tasks.values():
            if (len(records_for_task) != data["config"]["turns"]
                    or any(r["error"] is not None for r in records_for_task)):
                continue
            task_seconds.append(sum(r["e2e"] for r in records_for_task)
                                + data["config"]["turn_gap"] * (len(records_for_task) - 1))
            task_good += all(r["ttft"] <= data["config"]["ttft_slo"]
                             and (r["tpot"] is None or r["tpot"] <= data["config"]["tpot_slo"])
                             for r in records_for_task)
        s = trial["summary"]
        rows.append({"case": directory.name, "repetition": rep, "policy": policy,
                     "streams": data["config"]["streams"], "max_running": data["config"]["max_running"],
                     "requests": len(records), "errors": s["errors"], "wall": s["wall"],
                     "ttft_p50": s["ttft_p50"], "ttft_p90": s["ttft_p90"],
                     "e2e_p50": s["e2e_p50"], "tpot_p50": s["tpot_p50"],
                     "tpot_p90": s["tpot_p90"], "token_interval_p99": s["token_interval_p99"],
                     "coalesced_tokens": s["coalesced_tokens"],
                     "derived_task_p50": statistics.median(task_seconds) if task_seconds else None,
                     "goodput": s["goodput"], "slo_attainment": s["slo_attainment"],
                     "task_slo_attainment": task_good / data["config"]["streams"],
                     "copy_bytes": trial["preparation"].get("bytes", 0),
                     "preparation_wall": trial["preparation_wall"],
                     "group_init": data["group_init_seconds"]})
    return data, rows, requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# Skew readiness 结果复算", "",
             "每格为各重复的中位数；请求延迟包含准备等待。Task 是固定后缀、逐分支闭环回放，"
             "完成时间由逐轮 E2E 与固定 turn gap 求和，省略响应结束后的少量客户端记账时间。", ""]
    all_rows = []
    for directory in args.directories:
        data, rows, records = analyze(directory)
        all_rows.extend(rows)
        lines += [f"## {directory.name}", "",
                  f"前缀 {data['config']['prefix']}，分支 {data['config']['streams']}，"
                  f"轮数 {data['config']['turns']}，输出 {data['config']['output_tokens']}，"
                  f"单引擎并发上限 {data['config']['max_running']}。", "",
                  "| 策略 | 重复 | wall s | TTFT p50 s | TTFT p90 s | Task p50 s | TPOT p50 ms | goodput req/s | 联合 SLO |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        by_policy = defaultdict(list)
        for row in rows:
            by_policy[row["policy"]].append(row)
        for policy, group in by_policy.items():
            def median(field):
                return statistics.median(r[field] for r in group)
            lines.append(f"| {policy} | {len(group)} | {median('wall'):.3f} | "
                         f"{median('ttft_p50'):.3f} | {median('ttft_p90'):.3f} | "
                         f"{median('derived_task_p50'):.3f} | {median('tpot_p50') * 1000:.2f} | "
                         f"{median('goodput'):.3f} | {median('slo_attainment'):.1%} |")
        copies = by_policy.get("prepared_copy", [])
        comparisons = []
        for copy in copies:
            baselines = [r for r in rows if r["repetition"] == copy["repetition"]
                         and r["policy"] in ("sticky", "balanced", "cache_cost")]
            if baselines:
                best = min(baselines, key=lambda r: r["wall"])
                comparisons.append({"rep": copy["repetition"], "baseline": best["policy"],
                                    "wall_reduction": 1 - copy["wall"] / best["wall"],
                                    "exposed_init_wall": copy["wall"] + copy["group_init"],
                                    "baseline_wall": best["wall"]})
        lines += ["", f"原始记录 {len(records)} 条，完成试验 {len(rows)} 个。",
                  f"单独建链耗时 {data['group_init_seconds']:.3f}s，未计入服务窗口。"]
        if comparisons:
            lines += [f"复制相对每次重复中最快不准备策略的 wall 降幅："
                      + ", ".join(f"r{x['rep']} {x['wall_reduction']:.1%} ({x['baseline']})" for x in comparisons) + "。",
                      "如果建链完全暴露、只服务一次事件，复制窗口另加建链时间后分别为："
                      + ", ".join(f"{x['exposed_init_wall']:.2f}s vs {x['baseline_wall']:.2f}s" for x in comparisons) + "。"]
        reference = {(r["repetition"], r["consumer"], r["turn"]): r["output_sha256"]
                     for r in records if r["policy"] == "sticky" and r["error"] is None}
        for policy in by_policy:
            if policy == "sticky":
                continue
            comparable = [r for r in records if r["policy"] == policy and r["error"] is None
                          and (r["repetition"], r["consumer"], r["turn"]) in reference]
            equal = sum(r["output_sha256"] == reference[r["repetition"], r["consumer"], r["turn"]]
                        for r in comparable)
            lines.append(f"{policy} 与对应 sticky 的输出摘要一致：{equal}/{len(comparable)}。")
        lines.append("")
    if all_rows:
        with (args.out / "trials.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    (args.out / "REPORT.zh.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
