#!/bin/bash
#SBATCH --time=0-20:00:00
#SBATCH --job-name=arxiv_model
#SBATCH --output=arxiv_model.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --account=trz3akp

# copy and modify set_local_var.sh file
source gitignore_set_local_var

normalize_text() {
  awk '{print tolower($0);}' | tr -cd '[:print:]' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  -e 's/«/ /g' | tr 0-9 " "
}


cd $EMBEDBINFILES/word2vec
#DATA="/pylon5/ms5pi8p/lab232/miniclean"
#MODEL="/pylon5/ms5pi8p/lab232/model"`date '+%H-%M_%d-%m'`
#DATA="/home/luis/Documents/arxivDownload/embed/normText4NER"
#MODEL="/home/luis/rm_me/model4ner_"`date '+%H-%M_%d-%m'`

MODEL=$MODELDIR/"model_"`date '+%H-%M_%d-%m'`


mkdir $MODEL
cd $ARXIVDOWNDIR/embed
for file in $CLEANDATADIR/math*;
do
    echo "Concatenating file $file"
    cat $file | python3 run_normalize.py embed4classif >> $MODEL/data.txt
done


cd $EMBEDBINFILES/word2vec

#./word2phrase -train data.txt -output data-phrase.txt -threshold 200 -debug 2
#./word2phrase -train data-phrase.txt -output data-phrase2.txt -threshold 100 -debug 2
./word2vec -train $MODEL/data.txt -output $MODEL/vectors.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 10 -binary 1 -iter 2 -min-count 10000


