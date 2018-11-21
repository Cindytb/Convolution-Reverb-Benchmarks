#!/bin/bash
#SBATCH --gres=gpu:1
cd GPU
./princeBuild.sh
cd ../Sequential
./build.sh
cd ..
./buildPrecisionTest.sh

./precisionTime.out Audio/96000/2e10.wav Audio/96000/2e09.wav
