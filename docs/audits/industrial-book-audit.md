# Industrial book audit: first bounded revision

Baseline `c6da37312d62dedb64651bfbdc51ff81c76b2fe5`; inspected 2026-09-11.
The baseline has 15 chapter files (12,631 lines), 12 bibliography entries,
PDF-only CI and no executable code/test package. It is a strong introductory
spine, not an implemented industrial trainer.

This is an architecture/coverage audit with targeted numerical and code review,
not a claim that every sentence or historical experiment has been independently
reproduced. Full chapter-by-chapter industrial regeneration remains gated.
All original files are retained; no 117-chapter empty scaffold is created.

P = present introductory coverage; E = needs executable expansion;
R = needs source/provenance refresh; N = not this chapter's responsibility.
The matrix describes the baseline, before Part V.

| Chapter | Pedagogy | Math | Code | Systems | Industrial depth | Figures | Tests | Research | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Foundations roadmap | P | P | E | P | E | TikZ | absent | R | Preserve entry point; cross-reference four planes |
| Mathematical foundations | P | P/R | E | N | E | TikZ | absent | R | Repair weighted sum and unmasked label |
| Text/data/tokenization | P | P | E | P | E | TikZ | absent | R | Repair window bound; test identity and shift |
| Tensors/PyTorch | P | P | E | P | E | TikZ | absent | R | Expand autograd/precision later |
| Embeddings | P | P | E | P | E | TikZ | absent | R | Preserve learned positions; distinguish RoPE |
| Attention/transformers | P | P | E | P | E | TikZ | absent | R | Add causal GQA oracle in modern core |
| GPT architecture | P | P | E | P | E | TikZ | absent | R | Preserve TinyGPT; explicit lineage table |
| Loss/optimization | P | P | E | P | E | TikZ | absent | R | Connect to optimizer/accumulation tests |
| Pretraining engineering | P | P | E | P | E | TikZ | absent | R | Later data provenance and recovery |
| Evaluation/inference | P | P | E | P | E | TikZ | absent | R | Keep foundations; deep serving in companion |
| Model files/runtime buffers | P | P | E | P | E | TikZ | absent | R | Expand resume/export distinction |
| Hardware/backends | P | P | E | P | E | TikZ | absent | R | Relabel unverified historical throughput |
| Fine-tuning/projects | P | P | E | P | E | TikZ | absent | R | Later executable SFT/LoRA/preferences |
| Lab manual | P | P | E | P | E | TikZ | absent | R | Keep; add Part V executable labs |
| Glossary | P | P | N | P | E | tables | absent | R | Expand with verified concepts |

## Corrections

- Exclusive batch bound `N-T-1` corrected to `N-T`; smallest valid stream tested.
- Weighted sum corrected to `(0.405, 0.300, 0.305)` and explicitly unmasked.
- GPT21 throughput marked REPORTED inputs / DERIVED ratio: raw logs and run
  revision are missing, so it is not a newly reproduced benchmark.
- Wide TikZ diagrams receive a vector-preserving maximum-width guard; four
  new native plates show planes, block semantics, lifetimes and checkpoint cut.

Part V supplies the first executable modern core and local recovery contract.
Existing multi-backend excerpts are not three newly verified implementations.
