#!/bin/bash
#SBATCH --time=0-05:00:00
#SBATCH --job-name=train_glove
#SBATCH --output=glv_%j.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --account=dms200016p


# copy and modify set_local_var.sh file
source gitignore_set_local_var
SOURCE_DIR=`pwd`

#DATA="/pylon5/ms5pi8p/lab232/miniclean"
#MODEL="/pylon5/ms5pi8p/lab232/model"`date '+%H-%M_%d-%m'`
#DATA="/media/hd1/clean_text"
#DATA="/media/hd1/processed_text/joined_math20-02_24-01"
#MODEL="/home/luis/rm_me/model"`date '+%H-%M_%d-%m'`

MODEL=$MODELDIR/"glove_model_"`date '+%H-%M_%d-%m'`

mkdir -p $MODEL
echo "Using the parameters:" | tee -a $MODEL/log.txt
cat $SOURCE_DIR/gitignore_set_local_var | tee -a $MODEL/log.txt


#for file in $CLEANDATADIR/math*;
#do
#    echo "Concatenating file $file"
#    cat $file | python3 run_normalize.py embed4classif >> $MODEL/data.txt
#done
# USE A CLEAN FILE DIRECTLY INSTEAD OF SPENDING TIME CLEANING math*
echo "Using file $CLEANDATADIR/data.txt" | tee -a $MODEL/log.txt
cp $CLEANDATADIR/data.txt $MODEL/data.txt

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

echo "starting running bin files: " | tee -a $MODEL/log.txt
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $MODEL/data.txt > $MODEL/vocab.txt" \
    | tee -a $MODEL/log.txt
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $MODEL/data.txt > $MODEL/vocab.txt

echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $MODEL/vocab.txt -verbose $VERBOSE -window-size $WINDOW_SIZE < $MODEL/data.txt > $MODEL/cooccurrence.bin" | tee -a $MODEL/log.txt
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $MODEL/vocab.txt -verbose $VERBOSE -window-size $WINDOW_SIZE < $MODEL/data.txt > $MODEL/cooccurrence.bin

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $MODEL/cooccurrence.bin > $MODEL/cooccurrence.shuf.bin" \
    | tee -a $MODEL/log.txt
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $MODEL/cooccurrence.bin > $MODEL/cooccurrence.shuf.bin

echo "$ $BUILDDIR/glove -save-file $MODEL/vectors -threads $NUM_THREADS -input-file $MODEL/cooccurrence.shuf.bin -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $MODEL/vocab.txt -verbose $VERBOSE" \
    | tee -a $MODEL/log.txt
$BUILDDIR/glove -save-file $MODEL/vectors -threads $NUM_THREADS -input-file $MODEL/cooccurrence.shuf.bin -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $MODEL/vocab.txt -verbose $VERBOSE

#echo "$ $PYTHON eval/python/evaluate.py" | tee -a $MODEL/log.txt
#$PYTHON $EMBEDBINFILES"/GloVe/eval/python/evaluate.py" | tee -a $MODEL/log.txt
