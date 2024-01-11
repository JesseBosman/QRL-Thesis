#!/bin/bash
#SBATCH --job-name=run_PQC
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1000M

#SBATCH --partition="cpu-long"
#SBATCH --time=167:00:00
#SBATCH --ntasks=4
#SBATCH	--cpus-per-task=10

echo "#### starting PQC "
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
nh=6
Rx=1
echo "[$SLURM_JOB_NODELIST], [$SLURM_NTASKS]"

# set 1
for lr in 0.1 0.01 0.001
    do
    for n_eps in 2500 11600 53860 250000
    do
        for nl in {1..4}
        do
            python3 main_PQC.py -ls $ls  -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -ne $n_eps -nh $nh -ms $ms -bs 10 -en $env_name -li $lr -lv $lr -lo $lr -nl $nl -Rx $Rx -nr 10 &
        done
        wait

        for nl in {5..8}
        do
            python3 main_PQC.py -ls $ls  -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -ne $n_eps -nh $nh -ms $ms -bs 10 -en $env_name -li $lr -lv $lr -lo $lr -nl $nl -Rx $Rx -nr 10 &
        done
        wait

    done

done

for nl in {9..10}
do
    for lr in 0.1 0.01 0.001
    do
        for n_eps in 2500 11600 53860 250000
        do
            python3 main_PQC.py -ls $ls -b1 $b1 -t1 $t1 -b2 $b2 -t2 $t2 -ne $n_eps -nh $nh -ms $ms -bs 10 -en $env_name -li $lr -lv $lr -lo $lr -nl $nl -Rx $Rx -nr 10 &
        done
        wait
    done
done
        


echo "[$SHELL] ## Finished"

