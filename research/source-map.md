# Source and claims ledger

Reviewed 2026-09-11. Papers motivate semantics; tests establish local behavior.
Paper speedups are not adopted as our measurements.

| Primary source | Pin / use |
| --- | --- |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | 2019; shared RMS and gain, no recentering |
| [RoFormer](https://arxiv.org/abs/2104.09864) | 2021; Q/K rotations; adjacent-pair layout explicitly local |
| [GQA](https://arxiv.org/abs/2305.13245v3) | v3; grouped query-to-KV heads |
| [GLU variants](https://arxiv.org/abs/2002.05202) | 2020; gated MLP family |
| [AdamW](https://arxiv.org/abs/1711.05101) | ICLR 2019; decoupled weight decay |
| [PyTorch AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html) | Tested 2.8.0; foreach=False; scalar first-step oracle |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html) | Same-version CPU scope, no cross-platform guarantee |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | 2022; roadmap for compute/data budgets |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 2019; roadmap, not implemented TP |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 2020; roadmap, not implemented sharding |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022; roadmap; current scores are quadratic |

**MEASURED** means an actual run with command, runtime, seed, workload and result.
**DERIVED** means a formula with units/assumptions. **REPORTED** means a source's
observation with provenance limitations. **ESTIMATED** means a forecast.

Historical wish01/GPT21 totals are REPORTED with missing logs/revision, not
verified performance. No vendor benchmarks or model downloads were executed.
FSDP2, PyTorch distributed checkpoint, NCCL, Triton and Transformer Engine need
version-pinned source inspection at their implementation milestone; this reading
queue does not establish support.
