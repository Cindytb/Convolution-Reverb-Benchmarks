#!/bin/bash
#SBATCH --gres=gpu:1
host=$(hostname)
if [[ "$host" = *"cims"* ]]
    then host="cims"
    else host="prince"
fi

DIRS=" -L/share/apps/libsndfile/1.0.28/intel/lib \
-L/share/apps/cuda/9.2.88/lib64 \
-L/share/apps/fftw/3.3.6-pl2/intel/lib \
-L/home/ctb335/Capstone/Sequential \
-L/home/ctb335/Capstone/GPU
"

INC="-I/home/ctb335/libsndfile/include \
-I/home/ctb335/cuda/inc \
-I/share/apps/libsndfile/1.0.28/intel/include \
-I/share/apps/cuda/9.2.88/include \
-I/share/apps/cuda/9.2.88/samples/common/inc \
-I/share/apps/fftw/3.3.6-pl2/intel/include
"
LIBS="-lfftw3f -lm -lcufft_static -lculibos -lsndfile -lnvToolsExt "

if [[ $host = "cims" ]]; then
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 "
    module purge
    module load gcc-5.2.0
    module load cuda-9.1
else
     ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
     -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60\
      -gencode arch=compute_70,code=sm_70 "
    module purge
    module load gcc/6.3.0
    module load cuda/9.2.88
    module load libsndfile/intel/1.0.28
    module load fftw/intel/3.3.6-pl2
fi

if [[ "masterTest.cpp" -nt "masterTest.o" ]]; then
    echo Compiling masterTest
    nvcc ${ARCH} $INC -c masterTest.cpp
fi
echo Linking precision.out
nvcc -ccbin g++ masterTest.o \
    ${ARCH} ${DIRS} ${LIBS} -lseqconvolve -lgpuconvolve  \
    -o precision.out