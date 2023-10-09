#!/bin/bash
#SBATCH --job-name=run_PQC
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=500M

#SBATCH --partition="cpu-long"
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH	--cpus-per-task=1

echo "#### starting PQC "
echo "This is $SLURM_JOB_USER and my the job has the ID $SLURM_JOB_ID"
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and i am currently in $CWD"

echo "[$SHELL] ## Run script"
python3 main_PQC.py
echo "[$SHELL] ## Finished"

