#!/bin/bash
if [ "$#" -lt 3 ]; then
        echo "Usage: ./test.sh resultslog tempLog exportLog [time] [output]"
        exit 1
fi
hostName=$(hostname)
TIME=""
OUT=""
TAG="- GPU - FFT Convolution"
if [[ $4 == "time" ]]; then
    TIME="-t"
    TAG="- GPU - Time Domain Convolution"
fi
if [[ $4 == "output" || $5 == "output" ]]; then
    OUT="-o ${hostname}.wav"
fi
resultsLog=$1
tempLog=$2
exportLog=$3
>$resultsLog
>$tempLog
>$exportLog
for (( j=0; j<10; j=j+1 )); do
        echo ------------------------------------------------------------------------------- >> $resultsLog
        echo Iteration $j >> $resultsLog
        echo ------------------------------------------------------------------------------- >> $resultsLog
        echo -------------------------------------------------------------------------------
        echo Iteration $j
        echo -------------------------------------------------------------------------------

        for i in ../Audio/96000/*; do
                echo $i $TAG
                (time bin/gpuconvolve.out $TIME -i $i $OUT) >> $resultsLog 2>&1
                tail -3 $resultsLog  | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog
        done
        echo  >> $tempLog
done
python3 parse.py $tempLog $exportLog
echo -----------------------------------------------------------------------------------------
echo COMPLETED
echo -----------------------------------------------------------------------------------------
