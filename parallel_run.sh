#!/bin/bash
#SBATCH --job-name=parallel_latexml
#SBATCH --nodes=4
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --time=00:20:00 # 6 days walltime in dd-hh:mm format
#module purge #make sure the modules environment is sane
# Set a trap to copy any temp files you may need
srun $(time parallel -P 95% ./run_latexml.sh ::: $SCRATCH/large_test/math05/*)
