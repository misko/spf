# Build main.pdf. Requires texlive (e.g. apt install texlive-latex-extra).
all: main.pdf

main.pdf: main.tex sections/*.tex
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex   # second pass: TOC + refs

clean:
	rm -f main.aux main.log main.out main.toc main.pdf

.PHONY: all clean
