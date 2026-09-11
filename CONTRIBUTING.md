# Contributing

This repository contains the LaTeX source for the MapleAI LLM training textbook.

## Build

```bash
make dist
```

The generated PDF is written to:

```text
build/maple_gpt_textbook.pdf
```

Build output is intentionally ignored by git. Commit source changes in `main.tex`,
`chapters/`, `figures/`, `references.bib`, and project metadata.

## Writing Guidelines

- Keep chapters undergraduate-friendly: define terms before using them.
- Map math to the TinyGPT or MapleGPT training case whenever possible.
- Use native TikZ vector illustrations with meaningful arrows and readable labels;
  do not publish character-box graphs. Follow `figures/style.tex`.
- Keep examples runnable on small hardware first.
- Cite public references through `references.bib`.

## Review Checklist

- `make dist` completes.
- New figures fit on the rendered page and do not overlap labels or arrows.
- Code snippets are short enough to wrap cleanly.
- New references appear in the bibliography.
- `make check` and `make smoke` pass with the pinned CPU dependencies.
- Quantitative claims are labeled MEASURED, DERIVED, REPORTED or ESTIMATED.
- Preserve the TinyGPT/TinyLM3 distinction and training/inference ownership boundary.
