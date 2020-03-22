#!/bin/bash
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

NEW_NAME="math12"
#SOURCE_DIR=$SCRATCH/"11_tars"
SOURCE_DIR="/mnt/arXiv_src/src"

[ -z $RAMDISK ] && RAMDISK=/tmp
MAIN_DIR=$RAMDISK/$NEW_NAME # temporary store for speed

[ -z $SCRATCH ] && SCRATCH=$HOME;
OUT_DIR=$SCRATCH/$NEW_NAME;  # where the files should end up
mkdir -p $SCRATCH/$NEW_NAME;
START_HOME=$PWD
#cp $SCRATCH/arxiv2.db $RAMDISK/;


#for a in `ls $SOURCE_DIR`; do
for a in `ls $SOURCE_DIR/arXiv_src_1209_0{03,04,05,06,07,08,09,10}.tar | xargs -n 1 basename`; do
#for a in `ls $SOURCE_DIR/arXiv_src_12* |\
#    xargs -n 1 basename |\
#    awk 'BEGIN {FS="_"} {if ($3 > 1207 && $3 < 1210) print $0}'`; do
#for a in "arXiv_src_1112_004.tar"; do
# names of tar files have format:  arXiv_src_0508_001.tar 
# and naming the subdir 0508_001

SUBDIR_NAME=$(echo $a |  awk 'BEGIN {FS="[_.]"}; {print $3"_"$4}');
SUBDIR=$MAIN_DIR/$SUBDIR_NAME;
mkdir -p $SUBDIR;

python3 process.py $SOURCE_DIR/$a $SUBDIR --term math #--db $RAMDISK/arxiv2.db

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
[ -f paralatexml.log ] && cp paralatexml.log $SCRATCH/$NEW_NAME;
    }
trap run_on_exit EXIT