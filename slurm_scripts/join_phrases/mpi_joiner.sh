#!/bin/bash
#SBATCH --time=0-16:00:00
#SBATCH --job-name=joining_phrases
#SBATCH --output=log_phrases.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM

module load python/3.8.6
module load openmpi/3.1.6-gcc8.3.1
source $HOME/env1/bin/activate

OUTDATADIR="joined_math"`date '+%H-%M_%d-%m'`
MPILOOP="/jet/home/lab232/arxivDownload/MP_scripts/mpi_only_loop.py"

cd $HOME/arxivDownload/MP_scripts
mkdir -p $LOCAL/$OUTDATADIR

time mpirun --mca mpi_warn_on_fork 0 \
    python3 $MPILOOP $PROJECT"/promath/math*/*.tar.gz" \
    $LOCAL/$OUTDATADIR 2>&1

cp -r $LOCAL/$OUTDATADIR $PROJECT/$OUTDATADIR
