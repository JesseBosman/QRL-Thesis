#!/bin/bash
#SBATCH --job-name=run_NN
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1000M

#SBATCH --partition="cpu-short"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH	--cpus-per-task=10

echo "#### starting NN "
echo "This is $SLURM_JOB_USER and my the job has the ID $SLURM_JOB_ID"
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and i am currently in $CWD"

echo "[$SHELL] ## Run script"
source /home/s2025396/tfqenv/bin/activate
python3 main_NN.py
echo "[$SHELL] ## Finished"

