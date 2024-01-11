#!/bin/bash
#SBATCH --job-name=run_NN
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1000M

#SBATCH --partition="cpu-long"
#SBATCH --time=30:00:00
#SBATCH --ntasks=4
#SBATCH	--cpus-per-task=10

echo "#### starting NN "
echo "This is $SLURM_JOB_USER and my the job has the ID $SLURM_JOB_ID"
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and i am currently in $CWD"

echo "[$SHELL] ## Run script"

module purge
module load slurm
module load ALICE/default
module load Python/3.8.6-GCCcore-11.3.0
source /home/s2025396/tfqenv3/bin/activate

ls=5
env_name="givens"
ms=10
b1="gy"
t1=0.25
b2="gx"
t2=0.25
nh=7

for lr in 0.1 0.01 0.001
do
    for n_eps in 2500 11600 53860 250000
    do
        # set 1
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 1 -nnpl 2 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 1 -nnpl 3 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 1 -nnpl 4 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 1 -nnpl 5 -nr 10 &
        wait
        # #set 2
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 2 -nnpl 2 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 2 -nnpl 3 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 2 -nnpl 4 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 2 -nnpl 5 -nr 10 &
        wait
        # # set 3
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 3 -nnpl 2 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 3 -nnpl 3 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 3 -nnpl 4 -nr 10 &
        python3 main_NN.py -ls $ls -ne $n_eps -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -nh $nh -ms $ms -bs 10 -en $env_name -lr $lr -nhl 3 -nnpl 5 -nr 10 &
        wait
    done
done

echo "[$SHELL] ## Finished"