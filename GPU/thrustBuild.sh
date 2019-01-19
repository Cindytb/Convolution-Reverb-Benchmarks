#!/bin/bash
#SBATCH --gres=gpu:1
host=$(hostname)
if [[ "$host" = *"cims"* ]]; then 
    host="cims"
elif [[ "$host" = *"log"* ]] ; then 
    host="prince"
fi

WD=$(pwd)


OBJDIR="obj"
SRCDIR="src"

if [[ $host = "cims" ]]; then
    DIRS="-L/home/ctb335/Capstone/GPU"

    INC="-I/home/ctb335/libsndfile/include \
    -I/home/ctb335/cuda/inc
    "
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 "
    module purge
    module load gcc-5.2.0
    module load cuda-9.1
    
elif [[ $host = "prince" ]]; then
    DIRS="-L/share/apps/libsndfile/1.0.28/intel/lib \
    -L/share/apps/cuda/9.2.88/lib64
    -L/home/ctb335/Capstone/GPU
    "
    INC="-I/share/apps/libsndfile/1.0.28/intel/include \
    -I/share/apps/cuda/9.2.88/include \
    -I/share/apps/cuda/9.2.88/samples/common/inc 
    "
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70"
    module purge
    module load gcc/6.3.0
    module load cuda/9.2.88
    module load libsndfile/intel/1.0.28
else
    DIRS="-L/usr/lib/x86_64-linux-gnu\
    -L/usr/local/cuda-10.0/lib64
    -L$WD
    "
    INC="-I/usr/include \
    -I$WD/src/common/inc
    "
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70"
fi
echo Compiling thrustOps
nvcc -ccbin g++  $INC $DIRS -dc \
    $ARCH -DTHRUST_DEBUG \
    -o $OBJDIR/thrustOps.o -c $SRCDIR/thrustOps.cu
