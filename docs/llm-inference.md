# Cluster LLM Inference

Tiers:

| GPUs              | Partition  | Model                   |
| ----------------- | ---------- | ----------------------- |
| 1x A100 MIG 20 GB | `gpu_test` | Qwen3.5-9B int8 w8a8    |
| 1x A100 80 GB     | `seas_gpu` | Qwen3.8-27B bf16        |
| 1x H200           | `seas_gpu` | Qwen3.8-27B bf16        |
| 2x H200           | `seas_gpu` | undecided               |
| 4x H200, one node | `seas_gpu` | DeepSeek V4.1 Flash fp8 |
| more, multi node  |            | undecided               |

From the login node, run `make e2e-a100-20gb` for Qwen3.5-9B int8 or `make e2e-a100-80gb` for Qwen3.8-27B.

Start on the smallest tier that answers the question. Scale up only when it is the bottleneck.

Model selection:

- fits with kv headroom at the context you need
- vllm supports the arch and reasoning parser
- rank by agentic coding, then reasoning, then chat
- verify on the real hardware before committing

Sources:

- [LiveBench](https://livebench.ai/#/?cats=Coding%2CAgentic+Coding&open=1): agentic coding, primary signal. open weights, both coding categories
- [Onyx](https://onyx.app/open-llm-leaderboard): self-hostable only, carries params and vram
- [Vellum](https://www.vellum.ai/open-llm-leaderboard): swe-bench and terminal-bench, cross-check
- [Artificial Analysis](https://artificialanalysis.ai): cross-check
- [LMArena](https://lmarena.ai): human preference, weak for coding
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html): servable or not
- model card on Hugging Face: sampling, thinking controls, context, size

# Ranking

Ranks are among open weights.

| Model                      | Weights    | H200s | LiveBench cod | LiveBench agt | Onyx | Vellum |
| -------------------------- | ---------- | ----- | ------------- | ------------- | ---- | ------ |
| Kimi K3 bf16               | 1453.7 GiB | 11    | #1            | #1            | -    | -      |
| DeepSeek V4.1 Flash fp8 ⭐ | 475.3 GiB  | 4     | #3            | #1            | -    | -      |
| GLM-5.3 fp8                | 703.7 GiB  | 5     | #3            | #2            | -    | -      |
| GLM-5.3-Flash bf16         | 305.8 GiB  | 3     | #3            | #2            | -    | -      |
| Kimi K2.6 bf16             | 554.3 GiB  | 4     | #2            | #3            | #3   | #3     |
| MiniMax-M3 bf16            | 795.5 GiB  | 6     | -             | -             | #2   | #2     |
| Qwen3.8-27B bf16 ⭐        | 51.7 GiB   | 1     | #3            | #5            | -    | -      |
| Qwen3.6-27B bf16           | 51.7 GiB   | 1     | #6            | #6            | #7   | -      |
| Nemotron 3 Ultra NVFP4     | 328.1 GiB  | 3     | #6            | #7            | #16  | -      |
| Inkling NVFP4              | 551.3 GiB  | 4     | #5            | #8            | -    | -      |
| Motif 3                    | 586.4 GiB  | 5     | -             | -             | -    | -      |
