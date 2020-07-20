#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --job-name=svc_train
#SBATCH --output=svc_train.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=LM
#SBATCH --mem=250GB
#SBATCH --account=trz3aap


#CHECKLIST
# adjust SBATCH --time
# adjust number of samples


echo "starting job at "`date`
cd ../classifier_trainer
python3 trainer.py $HOME/training_defs/math1*/*.xml.gz\
    $HOME/rm_train_datalog\
    --samples 100000\
    --log debug
