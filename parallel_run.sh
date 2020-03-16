#!/usr/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --job-name=paraII
#SBATCH --output=paralatexml.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=6

#CHECKLIST
# time allocated to the script
# number of nodes
# name of new_name variable
# name of sourcedir variable
# list of files to extract
# term to extract

# MONTLY CHECKLIST
# job-name
# output log file name
# list of file to extract loop

echo "starting job at "`date`
source <(grep ^LATEXML_BIN "$PWD/config.toml")
echo "latexml_bin file is: $LATEXML_BIN"

NEW_NAME="math11"
SOURCE_DIR=$SCRATCH/"11_tars"
MAIN_DIR=$RAMDISK/$NEW_NAME # temporary store for speed
OUT_DIR=$SCRATCH/$NEW_NAME  # where the files should end up
mkdir -p $SCRATCH/$NEW_NAME
START_HOME=`pwd`
cp $SCRATCH/arxiv2.db $RAMDISK/;


#for a in `ls $SOURCE_DIR`; do
#for a in `ls $SOURCE_DIR/arXiv_src_1106* | xargs -n 1 basename`; do
for a in `ls $SOURCE_DIR | awk 'BEGIN {FS="_"} {if ($3 > 1106) print $0}'`; do
#for a in "arXiv_src_0808_002.tar" "arXiv_src_0808_003.tar"; do
# names of tar files have format:  arXiv_src_0508_001.tar 
# and naming the subdir 0508_001

SUBDIR_NAME=$(echo $a |  awk 'BEGIN {FS="[_.]"}; {print $3"_"$4}');
SUBDIR=$MAIN_DIR/$SUBDIR_NAME;
mkdir -p $SUBDIR;

python3 process.py $SOURCE_DIR/$a $SUBDIR --term math --db $RAMDISK/arxiv2.db

echo "Done extracting";

time parallel -P 95% ./run_latexml.sh ::: $SUBDIR/*;

echo "Taring files...";

cd $MAIN_DIR; #cd to get the right paths for tar
find $SUBDIR_NAME -name 'latexml*' -print0 -o -name '*.xml' -print0 |\
      tar -cf $RAMDISK/$SUBDIR_NAME.tar --null -T - ;
cd $START_HOME;

rm -r $SUBDIR;  #Clean Up
mv $RAMDISK/$SUBDIR_NAME.tar $SCRATCH/$NEW_NAME/; 


done;

echo "finished job at "`date`

run_on_exit() {
    cp paralatexml.log $SCRATCH/$NEW_NAME;
    }
trap run_on_exit EXIT
