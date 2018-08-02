* To check the files with with unknown encoding:
```bash
   find . -name 'commentary.txt' -exec grep Ignoring {} \; 
```
* To process the first .tex file to an .xml file of the same name and last part of error stream to commentary.txt
```bash
TEXF=`ls *.tex`; latexml $TEXF.tex 2>&1 > ${TEXF%.*}.xml | tail -15 >> commentary.txt
```
