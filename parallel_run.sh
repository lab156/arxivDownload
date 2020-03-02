#!/usr/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --job-name=paralatexml
#SBATCH --output=paralatexml
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=4

WORK_DIR="math05_001"
mkdir $SCRATCH/$WORK_DIR
python3 process.py $SCRATCH/large_test/arXiv_src_0501_001.tar $SCRATCH/$WORK_DIR/ --term math --db $SCRATCH/arxiv2.db

echo "Done extracting"

time parallel -P 95% ./run_latexml.sh ::: $SCRATCH/$WORK_DIR/*
