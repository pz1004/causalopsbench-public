# CausalOpsBench Benchmark Card

## Purpose

CausalOpsBench evaluates operational AI agents on dynamic incident response: detect an incident, explain its root cause, ground the explanation in evidence, select a constrained intervention, and improve final system state.

## Intended Uses

- Compare agentic, time-series, causal, and control systems under a shared interface.
- Stress-test cost-aware interventions rather than prose-only diagnosis.
- Provide a contamination-aware benchmark structure with public development seeds and hidden monthly test seeds.

## Non-Goals

- The initial scaffold is not a substitute for validated industrial simulators.
- The included synthetic episodes should be treated as development data, not final hidden benchmark data.
- The benchmark does not certify autonomous deployment safety.

## Recommended Tracks

- **Small:** 50 episodes, public synthetic domains, CPU-only baselines.
- **Medium:** 500 episodes, richer randomization, private seed refresh.
- **Full:** simulator-backed hidden tests, expert-audited causal graphs, standardized containers.

## Required Reporting

Submissions should report:

- model and agent framework versions,
- inference backend (for local runs: Ollama version and host),
- exact model tag and quantization/tag variant,
- hardware, including GPU model and available memory,
- context length (`num_ctx`), temperature, keep-alive, and thinking setting,
- prompt and tool manifest hashes,
- number of retries,
- token count, tool calls, wall-clock time, and approximate compute cost,
- JSON parsing failures and raw prediction artifacts,
- per-domain and aggregate scores,
- all safety violations.

## Contamination Controls

- Public development episodes use known seeds.
- Hidden tests should use unreleased seeds and generation timestamps.
- Episode IDs should be generated from seed/domain/index metadata without revealing the hidden seed.
- Leaderboard submissions should keep raw predictions for audit.
