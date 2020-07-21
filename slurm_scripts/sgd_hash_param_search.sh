#!/bin/bash
#SBATCH --time=0-40:00:00
#SBATCH --job-name=param_search
#SBATCH --output=param_search.log
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --nodes=1
#SBATCH --partition=LM
#SBATCH --mem=128GB
#SBATCH --account=trz3aap


#CHECKLIST
# adjust SBATCH --time
# adjust number of samples


echo "starting job at "`date`
cd ../classifier_trainer
python3 hash_sgd_param_search.py
