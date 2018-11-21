#!/bin/bash
#SBATCH --gres=gpu:2 -c2
#SBATCH --mem=62GB
if [ $# -eq 2 ]; then
	echo "Usage: $0 [freq|time|both]"
	exit 1;
fi
host=$(hostname)
if [[ "$host" = *"cims"* ]]
    then host="cims"
    else host="prince"
fi

RESULT=$1
STDERR=$1
RESULT+="$(hostname)_results.log"
STDERR+="$(hostname)_stderr.log"
>$RESULT
>$STDERR
WAV="/home/ctb335/Capstone/Audio/480000.wav"
if [[ $host = "cims" ]]; then
    module purge
    module load gcc-5.2.0
    module load cuda-9.1
else
    module purge
    module load gcc/6.3.0
    module load cuda/9.2.88
    module load libsndfile/intel/1.0.28
    module load fftw/intel/3.3.6-pl2
fi
if [ "$1"  == "time" ]; then
    echo RUNNING TIME PRECISION TEST
    for i in ~/Capstone/Audio/96000/*; do
        ./precision.out -i $i -r $WAV -t 1>>$RESULT 2>>$STDERR
    done
elif [ "$1"  == "freq" ]; then
    echo RUNNING FREQUENCY PRECISION TEST
    for i in ~/Capstone/Audio/96000/*; do
        ./precision.out -i $i -r $WAV 1>>$RESULT 2>>$STDERR
    done
elif [ "$1" == "both" ]; then
    echo RUNNING FREQUENCY PRECISION TEST
    for i in ~/Capstone/Audio/96000/*; do
        ./precision.out -i $i -r $WAV 1>>$RESULT 2>>$STDERR
    done
    echo
    echo
    echo
    echo
    echo
    echo
    echo RUNNING TIME PRECISION TEST
    for i in ~/Capstone/Audio/96000/*.wav; do
        ./precision.out -i $i -r $WAV -t 1>>$RESULT 2>>$STDERR
    done
else   
    echo "Usage: $0 [Freq|Time|Both]"   
fi

