# AgentShift

AgentShift makes a suspended-but-warm LLM agent mobile across serving engines.
At a completed-turn boundary it moves the immutable prefix KV, durably transfers
execution ownership, and overlaps the handoff with the tool-blocked interval.
The destination resumes with a full prefix-cache hit; epoch and step fencing
prevent the source and destination from advancing the same agent.

AgentShift is implemented as a Python control plane plus a patch against SGLang
commit `034dd39189ba1ace1308d3c8a58df275ef301a21`. The current implementation
supports homogeneous MHA engines with the same model revision and TP layout. It
does not migrate active decode requests or claim cross-node performance.

## Repository Layout

- `agentshift/`: durable continuation, ownership, placement, handoff, tiered
  baselines, and SGLang client code.
- `patches/agentshift-sglang.patch`: all SGLang data-plane changes relative to
  the pinned upstream commit.
- `scripts/`: end-to-end, baseline, workload, interference, elasticity, policy,
  and fault experiments.
- `tests/`: 39 control-plane and protocol tests.
- `results/`: JSON/CSV/Markdown experiment artifacts. Runtime SQLite databases
  are intentionally excluded.
- `paper_rewriting_output/`: paper sources, Arial-rendered figures, evidence
  audit, and the compiled 13-page USENIX PDF. Licensed Arial TTF source files
  are not redistributed.

## Quick Start

Detailed dependency, patch, server, test, and experiment commands are in
[`docs/reproduction.md`](docs/reproduction.md). The shortest validation path is:

```bash
python -m pip install -e '.[test]'
pytest -q

# With two patched SGLang engines already running on ports 31000 and 31001:
PYTHONPATH=. python scripts/smoke_e2e.py \
  --context-length 32768 \
  --agent-id smoke-32k
```

The smoke test fails unless the destination reports at least the complete
migrated prefix as cached.

## Main Results

- At 32K/500 ms, AgentShift matches Sticky and reduces post-tool latency by
  24.04x over rerouting and 2.41x over on-return migration.
- In an eight-agent 32K return burst, it moves 50% of owners with full hits and
  is 7.63x faster than rerouting.
- In an eight-agent, three-turn coding replay, it moves 50% of owners with 100%
  full hits and completes in 5.002 s, versus 7.169 s Sticky, 6.264 s reroute,
  and 5.318 s on-return.
- A real `DEST_READY` fault campaign passes 8/8 cases; source-shadow recovery is
  10.9x faster than cold reconstruction.

Literature-named Agentix/Autellix, Continuum, TokenCake, and Symphony baselines
are mechanism-equivalent implementations in the same SGLang testbed, not
official artifact reproductions.

## Documentation

- [System scope and invariants](docs/implementation_plan.md)
- [Reproduction guide](docs/reproduction.md)
- [Consolidated evaluation](docs/evaluation_report_20260721.md)
- [USENIX paper PDF](paper_rewriting_output/final_paper/paper.pdf)
- [Full Markdown paper](paper_rewriting_output/full_paper.md)

Validation status: 39/39 AgentShift tests, 14/14 SGLang prefix/TP tests,
PaperSpine PASS, and LaTeX guard 0 errors/0 warnings.
