.PHONY: build clean dist

PDF=build/main.pdf
DIST=build/maple_gpt_textbook.pdf

build:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

dist: build
	cp $(PDF) $(DIST)

clean:
	latexmk -C -outdir=build main.tex
	rm -rf build
