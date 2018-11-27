#!/bin/bash
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=ctb335@nyu.edu
#SBATCH --mem=62GB
#SBATCH --time=05:00:00
OUT=""
if [[ $1 == "output" ]]; then
    OUT="output"
fi
module purge
module load gcc/6.3.0
module load fftw/intel/3.3.6-pl2
module load libsndfile/intel/1.0.28
module load python3/intel/3.6.3

mkdir -p results
mkdir -p temp
mkdir -p exports
mkdir -p nohups
mkdir -p PIDs
hostNo=$(hostname)
./test.sh results/freqDomainResults${hostNo}.log temp/FDtemp${hostNo}.log exports/freqDomainExport${hostNo}.csv $OUT 