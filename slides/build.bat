SET OUT=slides

pandoc --slide-level 2 -V links-as-notes -V theme=bjeldbak -V aspectratio=169 --template=custom.beamer --toc -t beamer %OUT%.md -o %OUT%.pdf