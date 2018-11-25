#!/bin/bash
#SBATCH --gres=gpu:1
cd GPU
./build.sh noexec
cd ../Sequential
./build.sh noexec
cd ..
./buildPrecisionTest.sh

