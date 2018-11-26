#!/bin/bash
#SBATCH --gres=gpu:2 -c2
#SBATCH --mem=62GB
TIME=""
OUT=""
TAG="FD"
if [[ $1 == "time" ]]; then
    TIME="time"
    TAG="TD"
fi
if [[ $1 == "output" || $2 == "output" ]]; then
    OUT="output"
fi
module purge
module load gcc/6.3.0
module load cuda/9.2.88
module load libsndfile/intel/1.0.28
module load python3/intel/3.6.3
mkdir -p results
mkdir -p temp
mkdir -p exports
mkdir -p nohups
mkdir -p PIDs
hostNo=$(hostname)
./test.sh results/${TAG}results${hostNo}.log temp/${TAG}temp${hostNo}.log exports/${TAG}export${hostNo}.csv $TIME $OUT
