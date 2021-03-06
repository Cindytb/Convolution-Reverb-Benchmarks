#!/bin/bash
#SBATCH --mem=62GB
#SBATCH --time=24:00:00
TIME=""
OUT=""
if [[ $1 == "time" ]]; then
    TIME="time"
fi
if [[ $1 == "output" || $2 == "output" ]]; then
    OUT=""
fi
cd ~/Capstone/Sequential
mkdir -p results
mkdir -p temp
mkdir -p exports
mkdir -p nohups
mkdir -p PIDs

hostNo=$(hostname)
nohup ./test.sh $TIME $OUT results/freqDomainResults${hostNo}.log temp/FDtemp${hostNo}.log exports/freqDomainExport${hostNo}.csv &> nohups/freqDomainNohup${hostNo}.out &
echo $! > PIDs/$hostNo.pid
