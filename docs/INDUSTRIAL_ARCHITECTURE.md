# Industrial training architecture and milestone ladder

Keep Parts I–IV and their topic names. Part V begins an executable systems track.
Capabilities map into this spine, not into 117 empty chapter files.

| Plane / capability family | Home | Next correctness gate |
| --- | --- | --- |
| Math, tensors, differentiation | Foundations; tensors; modern core | Broader precision/autograd oracles |
| Data, tokenization, mixtures | Text/data; pretraining | Shard provenance and resumable sampling |
| Decoder, GQA, RoPE, SwiGLU | Model mechanics; modern core | Conversion equivalence |
| Optimizers and schedules | Loss/optimization; modern core | Scheduler/scaler state parity |
| Precision and memory | Hardware; checkpoint boundary | AMP reference and activation accounting |
| Kernels, fusion, compilation | Hardware/backends | Numerical gate before profiling |
| DP and state sharding | Distributed expansion | Two-process global-batch equivalence |
| TP, PP, CP, EP, collectives | Distributed expansion | Operator, routing and schedule oracles |
| Scaling and experiment design | Pretraining expansion | Budget model with units/assumptions |
| Checkpoints and operations | Checkpoint boundary | Durable recovery and failure injection |
| Telemetry and performance | Pretraining/hardware | Versioned manifests and measurements |
| Evaluation and post-training | Evaluation; fine-tuning | SFT/LoRA/preference objective tests |
| Security and governance | Operations expansion | Trusted artifacts, quotas, provenance |
| Training/inference co-design | Checkpoint boundary | Tokenizer/layout/logit export contract |

Correctness crosses model semantics, execution, distribution and operations.
An imported framework's capabilities are not proof of this repository's support.

1. **Implemented, CPU-tested:** modern model, gradients, optimizer, deterministic
   data and exact local optimizer-boundary continuation.
2. **Planned:** precision/memory, recomputation, overflow and accounting.
3. **Planned:** DDP reference, rank ownership and global-batch/resume equivalence;
   then optimizer/gradient/parameter sharding.
4. **Planned:** profiler-backed fusion, compilation, kernels and parallelism.
5. **Planned:** durable recovery, failure injection, provenance, telemetry/security.
6. **Planned:** training/inference artifact conversion and numerical equivalence.

The companion owns deep serving, request scheduling, KV/prefix memory,
speculation and SLAs. Training retains evaluation and export tradeoffs.
No converter currently connects TinyLM3 and mini-engine.
