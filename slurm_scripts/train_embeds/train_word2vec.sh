#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --job-name=arxiv_model
#SBATCH --output=w2v_%j.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM-shared
#SBATCH --account=dms200016p

# copy and modify set_local_var.sh file
source gitignore_set_local_var
SOURCE_DIR=`pwd`


cd $EMBEDBINFILES/word2vec
#DATA="/pylon5/ms5pi8p/lab232/miniclean"
#MODEL="/pylon5/ms5pi8p/lab232/model"`date '+%H-%M_%d-%m'`
#DATA="/home/luis/Documents/arxivDownload/embed/normText4NER"
#MODEL="/home/luis/rm_me/model4ner_"`date '+%H-%M_%d-%m'`

MODEL=$MODELDIR/"model_"`date '+%H-%M_%d-%m'`


mkdir $MODEL
echo "Using the parameters" > $MODEL/log.txt
cat $SOURCE_DIR/gitignore_set_local_var > $MODEL/log.txt

cd $ARXIVDOWNDIR/embed
#for file in $CLEANDATADIR/math*;
#do
#    echo "Concatenating file $file"
#    cat $file | python3 run_normalize.py embed4classif >> $MODEL/data.txt
#done
# USE A CLEAN FILE DIRECTLY INSTEAD OF SPENDING TIME CLEANING math*
echo "Using file $CLEANDATADIR/data.txt" | tee $MODEL/log.txt
cp $CLEANDATADIR/data.txt $MODEL/data.txt

echo "I ran on: $SLURM_NODELIST" | tee >> $MODEL/log.txt

cd $EMBEDBINFILES/word2vec

#./word2phrase -train data.txt -output data-phrase.txt -threshold 200 -debug 2
#./word2phrase -train data-phrase.txt -output data-phrase2.txt -threshold 100 -debug 2
./word2vec -train $MODEL/data.txt -output $MODEL/vectors.bin \
    -cbow $W2V_cbow \
    -size $W2V_size \
    -window $W2V_window \
    -negative $W2V_negative \
    -hs $W2V_hs  \
    -sample $W2V_sample \
    -threads $W2V_threads \
    -binary $W2V_binary \
    -iter $W2V_iter \
    -min-count $W2V_min_count
