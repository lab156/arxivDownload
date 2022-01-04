#!/bin/bash
#SBATCH --time=0-1:00:00
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

mkdir -p $LOCAL/OUTDATADIR

time mpirun python3 $MPILOOP $PROJECT/promath/math9{5,6,7}/*.tar.gz \
    $LOCAL/OUTDATADIR 2>$1

cp -r $LOCAL/$OUTDATADIR $PROJECT/$OUTDATADIR
