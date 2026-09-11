.PHONY: build clean dist test smoke check
PYTHON ?= python3

PDF=build/main.pdf
DIST=build/maple_gpt_textbook.pdf

build:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

dist: build
	cp $(PDF) $(DIST)
	mkdir -p output/pdf
	cp $(PDF) output/pdf/training-large-language-models.pdf

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=code $(PYTHON) code/examples/train_tiny.py
	PYTHONPATH=code $(PYTHON) code/examples/train_tiny.py --random-data

check: test
	$(PYTHON) scripts/check-book.py

clean:
	latexmk -C -outdir=build main.tex
	rm -rf build
