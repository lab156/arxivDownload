#!/bin/bash
#SBATCH --time=0-16:00:00
#SBATCH --job-name=joining_phrases
#SBATCH --output=log_phrases.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM


module load python/3.8.6
source $HOME/env1/bin/activate

cd $HOME/arxivDownload/embed
OUTDATADIR="joined_math"`date '+%H-%M_%d-%m'`
#MODEL="/pylon5/ms5pi8p/lab232/model"`date '+%H-%M_%d-%m'`


time python3 clean_and_token_text.py $PROJECT/clean_text/math* \
	$LOCAL/$OUTDATADIR \
	--phrases_file $PROJECT/glossary/v3/math*/*.xml.gz \
      	--num_phrases 30000 \
	2>&1

cp -r $LOCAL/$OUTDATADIR $PROJECT/$OUTDATADIR

