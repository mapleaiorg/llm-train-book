# Verification record: executable core milestone

2026-09-11, baseline `c6da37312d62dedb64651bfbdc51ff81c76b2fe5`.
Tested locally on macOS arm64, Python 3.9.6, torch 2.8.0, CPU FP32,
one intra-op thread and deterministic algorithms. No CUDA/MPS/distributed run.
Torch emits an optional missing-NumPy warning; the code does not use NumPy.

## Actual commands and results

`make check PYTHON=.venv/bin/python`: **17 tests passed**; TeX input, bibliography,
label/reference, figure linkage and corrected batching checks passed.
Tests include full-model finite differences, RMSNorm/RoPE gradcheck, an independent
head-by-head GQA oracle, prefix causality, unequal accumulation, scalar AdamW,
parameter-count and corrected book-arithmetic oracles, and exact resume.

`make smoke PYTHON=.venv/bin/python`, seed 42, 80 updates, default Config
(19,104 parameters, B=8, T=16, D=32, L=2, Hq=4, Hkv=2):

| Workload | Evaluation before | Evaluation after | Evidence |
| --- | ---: | ---: | --- |
| Repeating eight-character pattern | 2.308474064 | 0.008450269 | MEASURED; mechanics, not language generalization |
| Fresh independent uniform tokens | 2.222286463 | 2.087456465 | MEASURED; separate held-out RNG |
| Uniform entropy ln(8) | 2.079441542 | 2.079441542 | DERIVED, nats/token |

Exact resume compares uninterrupted updates 5–8 against restored updates 5–8:
losses, parameters, moments, step tensors, sampled-batch count and next RNG draw
match exactly on this runtime. This is not a portable bitwise guarantee.

## Publication and remaining gates

`make dist` builds the whole native LaTeX book with XeLaTeX/TeX Live 2025.
Two new chapters and four native TikZ plates extend the preserved teaching spine.
Vector width guards prevent old wide figures and tables from leaving the text block.
The 252-page vector-only PDF was rendered in full to overview sheets. The new
chapters and four plates were reviewed at full reading size. This caught and
fixed two label collisions in the lifetime/checkpoint illustrations. Long table
paths and a long chapter running header were also repaired. No undefined
citations/references or missing glyphs remain; minor baseline underfull spacing
warnings are not numerical or build failures.

Not validated: GPU precision/performance, DDP/FSDP/ZeRO, TP/PP/CP/EP, fused
attention, durable/distributed checkpoints, untrusted checkpoint security,
cross-book conversion or historical vendor/host benchmarks. Next milestone:
precision/memory correctness, then two-process update equivalence.
