#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --job-name=paraII
#SBATCH --output=paralatexml.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --account=trz3aap


#CHECKLIST
# adjust SBATCH --time
# number of nodes
# name of new_name variable
# name of sourcedir variable
# list of files to extract
# term to extract


echo "starting job at "`date`
cd ../classifier_trainer
python3 trainer.py $SCRATCH/training_defs/math1*/*.xml.gz\
    $HOME/rm_train_datalog\
    --samples 200\
    --log debug
