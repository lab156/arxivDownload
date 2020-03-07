#!/usr/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --job-name=paralatexml
#SBATCH --output=paralatexml.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1

MAIN_DIR="math05small"`date +%I-%M`
for a in 0502_00{1,2}; do
SUBDIR=$SCRATCH/$MAIN_DIR/$a;
mkdir -p $SUBDIR;
python3 process.py $SCRATCH/large_test/arXiv_src_$a.tar $SUBDIR --term math.AT --db $SCRATCH/arxiv2.db;

echo "Done extracting";

time parallel -P 95% ./run_latexml.sh ::: $SUBDIR/*;

#mv paralatexml.log $SUBDIR

done;
