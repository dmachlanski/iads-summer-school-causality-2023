pandoc --slide-level 2 -V links-as-notes -V theme=bjeldbak -V aspectratio=169 --pdf-engine=xelatex --template=custom.beamer -t beamer slides.md -o slides.pdf