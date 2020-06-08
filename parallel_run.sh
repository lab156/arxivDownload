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
# new_name variable
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
TAR_EXT="tar.gz"   # file extension of the output files
#cp $SCRATCH/arxiv2.db $RAMDISK/;


#for a in `ls $SOURCE_DIR`; do
#for a in `ls $SOURCE_DIR/arXiv_src_1404_0{11,12,13}.tar | xargs -n 1 basename`; do
#for a in `ls $SOURCE_DIR/arXiv_src_2003* | xargs -n 1 basename`; do
for a in `ls $SOURCE_DIR/arXiv_src_1212_* |\
    xargs -n 1 basename |\
    awk 'BEGIN {FS="[_.]"} {if ($4 > 0 && $4 < 3) print $0}'`; do
    #awk 'BEGIN {FS="[_.]"} {if ($3 > 0806 && $3 < 0810 && $4 > 0) print $0}'`; do
#for a in "arXiv_src_1110_011.tar"; do
# names of tar files have format:  arXiv_src_0508_001.tar 
# and naming the subdir 0508_001

SUBDIR_NAME=$(echo $a |  awk 'BEGIN {FS="[_.]"}; {print $3"_"$4}');
SUBDIR=$MAIN_DIR/$SUBDIR_NAME;
mkdir -p $SUBDIR;

#python3 process.py $SOURCE_DIR/$a $SUBDIR --term math #--db $RAMDISK/arxiv2.db
#python3 process.py $SOURCE_DIR/$a $SUBDIR --term math || exit 1;

#LOOP UNTIL process IS SUCCSESFUL
LOOP_SENTIN=1
while [ $LOOP_SENTIN -gt 0 ]
do
python3 process.py $SOURCE_DIR/$a $SUBDIR --term math 
LOOP_SENTIN=$?
if [ $LOOP_SENTIN -gt 0 ]
then
    echo "waiting a bit:" $LOOP_SENTIN;
    RN=$RANDOM; ((RN %= 20 )); ((RN += 10)); echo $RN
    echo "process.py failed, waiting $RN second, and cleaning up";
    rm -r $SUBDIR
    sleep $RN;
else
    echo "done processing:" $LOOP_SENTIN;
fi
done;


echo "Done extracting";

time parallel -P 95% ./run_latexml.sh ::: $SUBDIR/*;

echo "Taring files...";

cd $MAIN_DIR; #cd to get the right paths. tar needs this
OUTFILE_NAME=$SUBDIR_NAME.$TAR_EXT # ex. 0508_001.tar.gz

find $SUBDIR_NAME -name 'latexml*' -print0 -o -name '*.xml' -print0 |\
      tar -czf $RAMDISK/$OUTFILE_NAME --null -T - ;
cd $START_HOME;

rm -r $SUBDIR;  #Clean Up
mv $RAMDISK/$OUTFILE_NAME $SCRATCH/$NEW_NAME/; 

done;

echo "finished job at "`date`

run_on_exit() {
[ -f paralatexml.log ] && cp paralatexml.log $SCRATCH/$NEW_NAME;
    }
trap run_on_exit EXIT
