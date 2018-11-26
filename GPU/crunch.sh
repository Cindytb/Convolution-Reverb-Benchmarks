#!/bin/bash
#SBATCH --gres:gpu=2 -c2
#SBATCH --mem:62GB
TIME=""
OUT=""
if [[ $1 == "time" ]]; then
    TIME="time"
fi
if [[ $1 == "output" || $2 == "output" ]]; then
    OUT="output"
fi
mkdir -p results
mkdir -p temp
mkdir -p exports
mkdir -p nohups
mkdir -p PIDs
hostNo=$(hostname)
nohup ./test.sh $TIME $OUT results/freqDomainResults${hostNo}.log temp/FDtemp${hostNo}.log exports/freqDomainExport${hostNo}.csv &> nohups/freqDomainNohup${hostNo}.out &
echo $! > PIDs/$hostNo.pid
