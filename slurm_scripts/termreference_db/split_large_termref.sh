#!/bin/bash
#SBATCH --time=0-5:00:00
#SBATCH --job-name=split_termref
#SBATCH --output=S-%x.%j.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node=1
#SBATCH --mem=356

module load python/3.8.6
#module load openmpi/3.1.6-gcc8.3.1
source $HOME/env1/bin/activate

OUTDATADIR=/ocean/projects/mth220001p/lab232/termreference_db17-33_22-01

cd $HOME/arxivDownload/slurm_scripts/termreference_db
time python3 split_large_termref.py $OUTDATADIR  split_pickles \
    --termref term_ref_lst.pickle \
    -P 100 --prefix ter 2>&1

