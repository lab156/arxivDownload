#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --job-name=term_reference
#SBATCH --output=log_termref.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --account=mth220001p

module load python/3.8.6
#module load openmpi/3.1.6-gcc8.3.1
source $HOME/env1/bin/activate

OUTDATADIR="termreference_db"`date '+%H-%M_%d-%m'`

cd $HOME/arxivDownload/slurm_scripts/termreference_db
mkdir -p $LOCAL/$OUTDATADIR

time python3 make_db.py $PROJECT"/glossary/NN.v1/math*/*.xml.gz" \
    $LOCAL/$OUTDATADIR 2>&1

cp -r $LOCAL/$OUTDATADIR $PROJECT/$OUTDATADIR
