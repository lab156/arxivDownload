#!/usr/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --job-name=paralatexml
#SBATCH --output=paralatexml
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=4

time parallel -P 95% ./run_latexml.sh ::: $SCRATCH/large_test/math05/*
