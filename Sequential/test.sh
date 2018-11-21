#!/bin/bash
if [ "$#" -ne 3 ]; then
        echo "Usage: ./test.sh [time] resultslog tempLog exportLog"
        exit 1
fi

hostName=$(hostname)
TIME=""
OUT=""
if [[ $1 == "time" ]]; then
    TIME="-t"
fi
if [[ $1 == "output" || $2 == "output" ]]; then
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
                echo $i - CPU - FFT Convolution
                (time bin/seqConvolve.out $TIME -i $i $OUT) >> $resultsLog 2>&1
                tail -3 $resultsLog  | grep "real" | sed -e 's/^real[ \t]*//' | sed -e 's/s//' >> $tempLog
        done
        echo  >> $tempLog

done
python3 parse.py $tempLog $exportLog
echo -----------------------------------------------------------------------------------------
echo COMPLETED
echo -----------------------------------------------------------------------------------------
