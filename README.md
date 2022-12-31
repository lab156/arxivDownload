## Data Pipeline
### Downloading from arXiv
### Processing with LaTeXML
### More Processing
#### Getting Labeled Definitions
### Classifying Definitions
### NER
example with singularity:
```
singularity run --nv --bind $HOME/Documents/arxivDownload:/opt/arxivDownload,/media/hd1:/opt/data_dir 
    $HOME/singul/runner.sif python3 embed/inference_ner.py \
    --mine /opt/data_dir/glossary/inference_class_all/math96/*.xml.gz \
    --model /opt/data_dir/trained_models/ner_model/lstm_ner/ner_Sep-29_03-45/exp_001 \
    --out $HOME/rm_me_ner
```
#### Joining Phrases
* `MP_scripts/mpi_only_loop.py`
* `slurm_scripts/mpi_joiner.sh`

## Jupyter Notebooks
* Populating and examples SQLAlchemy databases
    * Filling the arxiv metadata database using `databases/create_db_define_models.py`
    * Query join examples in sqlalchemy query language
* Parsing Arxib Manifest and querying metadat.ipynb
    * Using magic module to find file info
    * Structure of the data in the manifest file
    * using the dload.py script and its objects
    * basic usage of the arxiv API package
    * very disorganized, mostly scratch work
* Time stats check output and logs.ipynb
    * code to read and interpret latexml log files
    * plot time of latexml processing
* getting problem articles for latexml.ipynb
    * Identify articles that are not included in the arxmliv database 
    * Try to process these problematic articles with either removing environments or with LaTeXTual
* Word embeddings generation and evaluation.py
    * read the binary files produced by word2vec
    * Get the raw text ready for embedders
    * Search for arxiv.db for the tags of an article
    * tSNE visualization of the tags of terms

## Scripts
* update_db.py
    * USAGE: python update_db.py DATABASE MANIFEST.xml tar_src_path [--log ]
    * Where database is a sqlite database and manifest is an xml file in the original format
    * tar_src_path is the dir where the tar files can be found
    * Ex. python3 update_db.py /mnt/databases/arxivDB.db ../arXiv_src_manifest_Oct_2019.xml /mnt/arXiv_src/
* process.py
    * Xtraction class reads and extracts a arXiv tar files.
    * Querying the arxiv metadata with the arxiv API and the arxiv.py package
    * Xtraction(tarfilename, db='sqlite:///pathdb') to read metadata from a database instead of api
    * Writing arxiv metadata to a database.


### Queries
* Index the article ID column to speedup queries
```sql
CREATE INDEX id_ind on articles(id);
```
To search and article, run with the following query:
```sql
select tags from articles where id between "http://arxiv.org/abs/{0}" and "http://arxiv.org/abs/{0}{{";
```
* Count the articles in a year of tar files
```sql
SELECT  count(articles.id) FROM manifest LEFT JOIN articles on manifest.id = articles.tarfile_id WHERE manifest.filename LIKE 'src/arXiv_src_06%' and articles.tags like '[{''term'': ''math%';
```
* Find the authors (in general) with the most publications
```sql
SELECT author, count(*) AS c FROM articles GROUP BY author ORDER BY c DESC LIMIT 10;
```
* Hack to find main article tag
```sql
 SELECT count(tags) FROM articles where tags LIKE '[{''term'': ''math.DG''%';
```
* find repeated entries where DataId is the repeated term
```sql
SELECT DataId, COUNT(*) c FROM DataTab GROUP BY DataId HAVING c > 1;
```
* Left join to quickly find all articles in a tar file
```sql
SELECT  articles.id, tags FROM manifest LEFT JOIN articles on manifest.id = articles.tarfile_id WHERE manifest.id = 1747;
```

* To check the files with with unknown encoding:
```bash
   find . -name 'latexml_commentary.txt' -exec grep Ignoring {} \;
```
* To process the first .tex file to an .xml file of the same name and last part of error stream to latexml_commentary.txt
```bash
TEXF=`ls *.tex`; latexml $TEXF.tex 2>&1 > ${TEXF%.*}.xml | tail -15 >> latexml_commentary.txt
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
* The distribution of the sizes of the tar files in the manifest:
```python
Counter({Interval(-1857373.906, 382162956.2, closed='right'): 273,
         Interval(382162956.2, 764272737.4, closed='right'): 2222,
         Interval(764272737.4, 1146382518.6, closed='right'): 3,
         Interval(1528492299.8, 1910602081.0, closed='right'): 1})
Large files
src/arXiv_src_1405_008.tar|805505033
src/arXiv_src_1512_003.tar|1910602081
src/arXiv_src_1812_033.tar|835663353
src/arXiv_src_1908_006.tar|803583004
```


### Definitions Tags
* ltx_theorem_df -- /math.0406533

### Problems
* LateXML did not finish 2014/1411.6225/bcdr_en.tex

### Testing
* All the tests in the ./tests directory are discovered with the command. Run
  from the repo directory
```
PYTHONPATH="./tests" python -m unittest discover -s tests
```
Or, from the `tests` directory, run:
```
PYTHONPATH=".." python -m unittest discover -s tests
```

The xml_file.xml is modified by the search.py module:
* *processed*, is False by default.
* *search* exists only when locate has been ran on the filesystem. It is true, when the file was found and False if the file _has_ been searched and not found.
