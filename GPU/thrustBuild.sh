#!/bin/bash
#SBATCH --gres=gpu:1
host=$(hostname)
if [[ "$host" = *"cims"* ]]
    then host="cims"
    else host="prince"
fi


OBJDIR="obj"
SRCDIR="src"
if [[ $host = "cims" ]]; then
    INC="-I/home/ctb335/cuda/inc"
    DIRS=""
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 "
    module purge
    module load gcc-5.2.0
    module load cuda-9.1
else
    INC="-I/share/apps/cuda/9.2.88/include \
    -I/share/apps/cuda/9.2.88/samples/common/inc "
    DIRS="-L/share/apps/cuda/9.2.88/lib64"
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70 "
    module purge
    module load gcc/6.3.0
    module load cuda/9.2.88
    module load libsndfile/intel/1.0.28
fi
echo Compiling thrustOps
nvcc -ccbin g++  $INC $DIRS -dc \
    $ARCH -DTHRUST_DEBUG \
    -o $OBJDIR/thrustOps.o -c $SRCDIR/thrustOps.cu
