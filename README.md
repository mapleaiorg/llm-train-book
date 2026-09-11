# Training Large Language Models From Scratch

*From tokens and gradients to industrial distributed training.*

September 2026 revision: the original 15 chapter files are retained. Part V adds
two LaTeX chapters: **An Executable Modern Training Core** and **The Checkpoint
Boundary**, with four native TikZ illustrations and a tested CPU training core.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make check PYTHON=.venv/bin/python
make smoke PYTHON=.venv/bin/python
make dist
```

The full PDF is `output/pdf/training-large-language-models.pdf`; the historical
alias below is retained. TinyLM3 implements RMSNorm, RoPE, causal GQA, SwiGLU,
AdamW, token-weighted accumulation and exact local CPU checkpoint continuation.
It is a correctness substrate, **not** a distributed or mixed-precision trainer.
The original TinyGPT lineage remains explicitly distinct.

See [audit](docs/audits/industrial-book-audit.md),
[architecture](docs/INDUSTRIAL_ARCHITECTURE.md),
[source ledger](research/source-map.md), and [verification](docs/VERIFICATION.md).
Companion: [Inside the LLM Engine](https://github.com/hermonai/inside-the-llm-engine/tree/astra-visual-rewrite),
*From model weights and KV memory to industrial inference serving.*
The books do not yet exchange a tested checkpoint format.

## Original teaching spine

This repository contains the standalone LaTeX textbook source for:

**Training Large Language Models From Scratch**

GitHub home:

```text
https://github.com/mapleaiorg/llm-train-book
```

The original teaching spine is organized by topic into four parts, followed now by the executable Part V:

- Foundations
- Model Mechanics
- Training
- Systems And Practice

Build:

```bash
make dist
```

The generated PDF is:

```text
build/maple_gpt_textbook.pdf
```

Build artifacts are ignored by git. The repository should track source, diagrams,
bibliography, and project metadata rather than generated LaTeX output.

Main source:

```text
main.tex
chapters/
references.bib
```

The content is undergraduate-oriented and combines conceptual explanation, math, engineering practice, and hands-on labs for training GPT-style language models from scratch.

## Repository Hygiene

- `main` is the default branch.
- CI builds the LaTeX source with XeLaTeX.
- Pull requests should include the build command used for verification.
- Diagrams should be reviewed in the rendered PDF, not only in source form.
