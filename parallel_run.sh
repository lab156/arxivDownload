#!/usr/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --job-name=paralatexml
#SBATCH --output=/tmp/paralatexml.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=4

#CHECKLIST
# time allocated to the script
#name of maindir variable
#number of nodes
# term to extract
echo "starting job at "`date`
source <(grep ^LATEXML_BIN "$PWD/config.toml")
echo "latexml_bin file is: $LATEXML_BIN"

MAIN_DIR=$SCRATCH/"math05"
for a in `ls $SCRATCH/large_test`; do
# Expecting names with format  arXiv_src_0508_001.tar 
# and naming the subdir 0508_001
SUBDIR=$MAIN_DIR/$(echo $a |  awk 'BEGIN {FS="[_.]"}; {print $3"_"$4}');
mkdir -p $SUBDIR;
python3 process.py $SCRATCH/large_test/$a $SUBDIR --term math --db $SCRATCH/arxiv2.db;

echo "Done extracting";

time parallel -P 95% ./run_latexml.sh ::: $SUBDIR/*;

done;

echo "finished job at "`date`

run_on_exit() {
    cp /tmp/paralatexml.log $MAIN_DIR
    }
trap run_on_exit EXIT
