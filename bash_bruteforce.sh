#!/bin/bash
#SBATCH --job-name=run_bruteforce
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1000

#SBATCH --partition="cpu-medium"
#SBATCH --time=14:00:00
#SBATCH --ntasks=3
#SBATCH	--cpus-per-task=1

echo "#### starting bruteforce "
echo "This is $SLURM_JOB_USER and my the job has the ID $SLURM_JOB_ID"
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and i am currently in $CWD"

echo "[$SHELL] ## Run script"

module purge
module load slurm
module load ALICE/default
module load Python/3.8.6-GCCcore-11.3.0
source /home/s2025396/tfqenv3/bin/activate
#  python3 main_NN.py -ls 4 -p1 $(bc <<< "scale = 10; 3./14.") -ne 250000 -nh 6 -ms 1000 -bs 10 -en "QFIAHv2" -lr 0.001 -nhl 1 -nnpl 2 -nr 10 &
python3 bruteforce_policy.py -N 10 -H 7 -E givens -b1 gx -t1 0.25 -b2 gy -t2 0.5 &
python3 bruteforce_policy.py -N 10 -H 7 -E givens -b1 gx -t1 1 -b2 gy -t2 0.5 &
python3 bruteforce_policy.py -N 10 -H 7 -E givens -b1 gy -t1 0.25 -b2 gx -t2 0.25 &
wait
echo "[$SHELL] ## Finished"

