"""Render the entire book into overview sheets plus full-size new chapter pages."""
from pathlib import Path
import subprocess
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
pdf = root / "output/pdf/training-large-language-models.pdf"
out = root / "build/qa"
out.mkdir(parents=True, exist_ok=True)
reader = PdfReader(pdf)
assert len(reader.pages) > 200
text = "\n".join(p.extract_text() or "" for p in reader.pages)
assert "An Executable Modern Training Core" in text
assert "The Checkpoint Boundary" in text
assert "\ufffd" not in text
assert not any(p.images for p in reader.pages), "unexpected raster image"
subprocess.run(["pdftoppm", "-scale-to", "600", "-png", str(pdf), str(out / "page")], check=True)
paths = sorted(out.glob("page-*.png"))
assert len(paths) == len(reader.pages)
for start in range(0, len(paths), 24):
    sheet = Image.new("RGB", (1800, 1800), "#dfe5ea")
    draw = ImageDraw.Draw(sheet)
    for offset, path in enumerate(paths[start:start+24]):
        tile = ImageOps.contain(Image.open(path).convert("RGB"), (280, 415))
        x, y = offset % 6 * 300 + 10, offset // 6 * 450 + 25
        sheet.paste(tile, (x,y))
        draw.text((x,y-18), path.stem, fill="black")
    sheet.save(out / f"overview-{start//24+1:02}.png")
for number,page in enumerate(reader.pages,1):
    page_text = page.extract_text() or ""
    # Match chapter opening, not the TOC entry.
    if "Chapter 16" in page_text and "An Executable Modern Training" in page_text:
        subprocess.run(["pdftoppm","-f",str(number),"-scale-to","1400","-png",str(pdf),str(out/"new")],check=True)
        print("New chapters begin at physical page",number)
        break
print("PDF review assets ready:",len(reader.pages),"vector-only pages")
