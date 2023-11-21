#!/bin/bash
#SBATCH --job-name=run_PQC_qibo
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1000M

#SBATCH --partition="cpu-medium"
#SBATCH --time=23:00:00
#SBATCH --ntasks=1
#SBATCH	--cpus-per-task=10

echo "#### starting PQC "
echo "This is $SLURM_JOB_USER and my the job has the ID $SLURM_JOB_ID"
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and i am currently in $CWD"

echo "[$SHELL] ## Run script"
deactivate
module purge
module load slurm
module load ALICE/default
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

source /home/s2025396/qibo/bin/activate
python3 main_PQC_qibo.py
echo "[$SHELL] ## Finished"

