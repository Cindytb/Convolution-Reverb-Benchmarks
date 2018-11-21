#!/bin/bash
TIME=""
if [[ $1 == "time" ]]; then
    TIME="time"
fi
mkdir -p results
mkdir -p temp
mkdir -p exports
mkdir -p nohups
mkdir -p PIDs
hostNo=$(hostname)
nohup ./test.sh $TIME results/freqDomainResults${hostNo}.log temp/FDtemp${hostNo}.log exports/freqDomainExport${hostNo}.csv &> nohups/freqDomainNohup${hostNo}.out &
echo $! > PIDs/$hostNo.pid
