"""Fail on dangling TeX inputs, citations, references or duplicate labels."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
sources = list((root / "chapters").glob("*.tex")) + [root / "main.tex"] + list((root / "figures").glob("*.tex"))
text = "\n".join(path.read_text() for path in sources)
for target in re.findall(r"\\input\{([^}]+)\}", text):
    assert (root / (target + ".tex")).is_file(), target
keys = set(re.findall(r"@\w+\{([^,]+),", (root / "references.bib").read_text()))
for group in re.findall(r"\\(?:no)?cite\{([^}]+)\}", text):
    for key in group.split(","):
        assert key in keys, key
labels = re.findall(r"\\label\{([^}]+)\}", text)
assert len(labels) == len(set(labels)), "duplicate labels"
for label in re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text):
    assert label in labels, label
for figure in ("training-planes", "modern-block", "step-lifetime", "checkpoint-cut"):
    assert "\\input{figures/" + figure + "}" in text
assert "high=len(tokens) - block_size," in text
assert "high=len(tokens) - block_size - 1," not in text
print(f"Book source checks passed: {len(sources)} TeX sources, {len(keys)} bibliography keys, {len(labels)} labels")
