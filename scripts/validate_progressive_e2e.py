from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


async def run(args) -> None:
    source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
    destination = SGLangAgentShiftClient("engine-b", args.destination, timeout=300)
    prompt = [args.prompt_salt] + [100] * (args.prefix_length - 1)

    async def run_case(*, progressive: bool, state_path: Path):
        await asyncio.gather(flush(source), flush(destination))
        store = SQLiteStateStore(state_path)
        coordinator = MigrationCoordinator(
            store,
            {"engine-a": source, "engine-b": destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=True,
        )
        case = "progressive" if progressive else "baseline"
        agent_id = f"validate-{case}"
        _, first = await timed_generate(
            source,
            prompt,
            max_new_tokens=args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        store.register_agent(
            AgentContinuation(
                agent_id,
                1,
                "engine-a",
                1,
                completed,
                f"tool-{agent_id}",
            )
        )
        continuation = tuple(
            list(completed) + [200] * args.tool_result_tokens
        )
        if progressive:
            result = await coordinator.migrate_and_generate_progressive(
                agent_id,
                "engine-b",
                token_ids=continuation,
                max_new_tokens=args.output_tokens,
                rid=f"{agent_id}-turn-2",
                layer_group_size=args.layer_group_size,
            )
            second = result.generation
        else:
            await coordinator.migrate(agent_id, "engine-b")
            _, second = await timed_generate(
                destination,
                list(continuation),
                max_new_tokens=args.output_tokens,
                rid=f"{agent_id}-turn-2",
            )
        return {
            "first_output_ids": first["output_ids"],
            "second_output_ids": second["output_ids"],
            "cached_tokens": int(second["meta_info"]["cached_tokens"]),
            "expected_cached_tokens": len(completed),
        }

    with tempfile.TemporaryDirectory(prefix="agentshift-progressive-") as temp_dir:
        root = Path(temp_dir)
        baseline = await run_case(progressive=False, state_path=root / "baseline.db")
        progressive = await run_case(
            progressive=True, state_path=root / "progressive.db"
        )

    first_match = baseline["first_output_ids"] == progressive["first_output_ids"]
    second_match = baseline["second_output_ids"] == progressive["second_output_ids"]
    full_hits = all(
        case["cached_tokens"] >= case["expected_cached_tokens"]
        for case in (baseline, progressive)
    )
    report = {
        "prefix_length": args.prefix_length,
        "layer_group_size": args.layer_group_size,
        "first_turn_match": first_match,
        "continuation_match": second_match,
        "full_prefix_hits": full_hits,
        "baseline": baseline,
        "progressive": progressive,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not (first_match and second_match and full_hits):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--prefix-length", type=int, default=32768)
    parser.add_argument("--prompt-salt", type=int, default=26000)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=771)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--layer-group-size", type=int, default=12)
    parser.add_argument("--transfer-port", type=int, default=30900)
    parser.add_argument("--tp-size", type=int, default=2)
    asyncio.run(run(parser.parse_args()))
