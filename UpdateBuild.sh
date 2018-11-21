#!/bin/bash
#SBATCH --gres=gpu:1
cd GPU
./build.sh
cd ../Sequential
./build.sh
cd ..
./buildPrecisionTest.sh

