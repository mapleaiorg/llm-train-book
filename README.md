# MapleGPT Textbook

This repository contains the standalone LaTeX textbook source for:

**Training Large Language Models From Scratch: An Undergraduate Textbook for Theory, Engineering, and Hands-on Practice**

Planned GitHub home:

```text
https://github.com/mapleaiorg/llm-train-book
```

The book has been rewritten as a logical textbook, not as a numbered `00-29` tutorial export. Chapter files are named by topic and organized into four parts:

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
