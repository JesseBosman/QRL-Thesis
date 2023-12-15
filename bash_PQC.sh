#!/bin/bash
#SBATCH --job-name=run_PQC
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="jesse.w.bosman@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=1500M

#SBATCH --partition="cpu-short"
#SBATCH --time=04:00:00
#SBATCH --ntasks=2
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
module load Python/3.8.6-GCCcore-11.3.0
source /home/s2025396/tfqenv3/bin/activate
# python3 main_PQC.py -ls 4 -p1 $(bc <<< "scale = 10; 3./14.") -ne 250000 -nh 6 -ms 1000 -bs 10 -en "QFIAHv2" -li 0.01 -lv 0.01 -lo 0.01 -nl 5 -Rx 1 -nr 10 &
# python3 main_PQC.py -ls 4 -p1 $(bc <<< "scale = 10; 3./14.") -ne 250000 -nh 6 -ms 1000 -bs 10 -en "QFIAHv2" -li 0.01 -lv 0.01 -lo 0.01 -nl 3 -Rx 1 -nr 10 &
# python3 main_PQC.py -ls 4 -p1 $(bc <<< "scale = 10; 3./14.") -ne 250000 -nh 6 -ms 1000 -bs 10 -en "QFIAHv2" -li 0.01 -lv 0.01 -lo 0.01 -nl 1 -Rx 1 -nr 10 &
# wait

python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 6 -ms 1000 -bs 10 -en "Givens" -li 0.1 -lv 0.1 -lo 0.1 -nl 5 -Rx 1 -nr 10 &
python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 6 -ms 1000 -bs 10 -en "Givens" -li 0.01 -lv 0.01 -lo 0.01 -nl 5 -Rx 1 -nr 10 &
wait

# python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 5 -ms 1000 -bs 10 -en "Givens" -li 0.01 -lv 0.001 -lo 0.01 -nl 2 -Rx 1 -nr 10 &
# python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 5 -ms 1000 -bs 10 -en "Givens" -li 0.01 -lv 0.001 -lo 0.01 -nl 3 -Rx 1 -nr 10 &
# python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 5 -ms 1000 -bs 10 -en "Givens" -li 0.01 -lv 0.001 -lo 0.01 -nl 4 -Rx 1 -nr 10 &
# python3 main_PQC.py -ls 2  -b1 "gx" -t1 0.25 -b2 "gy" -t2 0.5 -ne 250000 -nh 5 -ms 1000 -bs 10 -en "Givens" -li 0.01 -lv 0.001 -lo 0.01 -nl 5 -Rx 1 -nr 10 &
# wait

echo "[$SHELL] ## Finished"

