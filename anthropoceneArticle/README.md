# Instructions to Build the *Themes of the Anthropocene* Article via Pandoc 2

```bash
# 1. Install Pandoc
sudo apt-get install pandoc

# 2. Install Pandoc's LaTeX Dependencies
sudo apt-get install texlive-xetex

# Run pandoc with citeproc and pandoc-crossref
pandoc -s -o themes-of-the-anthropocene.pdf themes-of-the-anthropocene.md --filter pandoc-crossref --filter pandoc-citeproc --bibliography=bibliography.bib --csl=chicago-author-date.csl
```