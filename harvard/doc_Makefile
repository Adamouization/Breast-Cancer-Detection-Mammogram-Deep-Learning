# harvard family of bibliographic styles - documentation
# Copyright - Peter Williams peterw@archsci.arch.su.edu.au

all: harvard.ps

harvard.ps: harvard.dvi
	dvips harvard

harvard.dvi: harvard.tex harvard.bbl
	latex harvard
	latex harvard

harvard.bbl: harvard.aux harvard.bib
	bibtex harvard

harvard.aux: harvard.tex
	latex harvard

clean:
	rm -f harvard.dvi harvard.aux harvard.bbl harvard.log harvard.blg
