## Jupyter Notebooks
* Populating and examples SQLAlchemy databases
    * Filling the database with the arxiv metadata using `databases/create_db_define_models.py`

### Queries
* Find the authors (in general) with the most publications
```sql
SELECT author, count(*) AS c FROM articles GROUP BY author ORDER BY c DESC LIMIT 10;
```
* Hack to find main article tag
```sql
 SELECT count(tags) FROM articles where tags LIKE '[{''term'': ''math.DG''%';    
```

* To check the files with with unknown encoding:
```bash
   find . -name 'commentary.txt' -exec grep Ignoring {} \; 
```
* To process the first .tex file to an .xml file of the same name and last part of error stream to commentary.txt
```bash
TEXF=`ls *.tex`; latexml $TEXF.tex 2>&1 > ${TEXF%.*}.xml | tail -15 >> commentary.txt
```

* To find directories unprocessed by latexml (don't have a latexml_errors_mess.txt file)
```
find ./* -maxdepth 0 -type d '!' -exec test -e "{}/latexml_errors_mess.txt" ';' -print
```

* To filter manually cancelled latexml processes search in the latex_errors file with:
```
Fatal:perl:die Perl died
```

* When LaTeXML runs out of memory for example in 1504.06138
```
(Processing definitions /usOut of memory!
```


### Notes
* There is a limit of around 500 articles id that the API can handle.
* In 2014 the article name format changed from YYMM.{4 digits} to 5 digits.
* In March 2007, the naming format of the articles changed from 0701/math0701672 to 1503/1503.08375.

### Definitions Tags
* ltx_theorem_df -- /math.0406533

### Problems
* LateXML did not finish 2014/1411.6225/bcdr_en.tex


The xml_file.xml is modified by the search.py module:
* *processed*, is False by default.
* *search* exists only when locate has been ran on the filesystem. It is true, when the file was found and False if the file _has_ been searched and not found.
