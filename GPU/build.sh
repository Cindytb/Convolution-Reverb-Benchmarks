#!/bin/bash
#SBATCH --gres=gpu:1
host=$(hostname)
if [[ "$host" = *"cims"* ]]; then 
    host="cims"
elif [[ "$host" = *"log"* ]] ; then 
    host="prince"
fi

WD=$(pwd)
CC="gcc"
LIBS="-lm -lcufft_static -lculibos -lsndfile -lnvToolsExt "
CSRC=()
CUSRC=("Main" "sgDevConvolution" "Convolution" "MemManage" "FFT" "Audio" "timeDomain" "runner" )
#Not compiling ThrustOps or runner because it never seems to change
#FLAGS="-Xptxas=-v"
BINDIR="bin"
OBJDIR="obj"
SRCDIR="src"
mkdir -p $BINDIR
mkdir -p $OBJDIR

if [[ $host = "cims" ]]; then
    DIRS="-L$WD"

    INC="-I/home/ctb335/libsndfile/include \
    -I$WD/src/common/inc
    "
    ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 "
    #ARCH="-gencode arch=compute_60,code=sm_60 "
    module purge
    module load gcc-5.2.0
    module load cuda-9.1
    
elif [[ $host = "prince" ]]; then
    DIRS="-L/share/apps/libsndfile/1.0.28/intel/lib \
    -L/share/apps/cuda/9.2.88/lib64 \
    -L/share/apps/gromacs/5.1.4/openmpi/intel/lib64 \
    -L$WD
    "
    INC="-I/share/apps/libsndfile/1.0.28/intel/include \
    -I/share/apps/cuda/9.2.88/include \
    -I$WD/src/common/inc
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
    #ARCH="-gencode arch=compute_60,code=sm_60 "
    CC="gcc-7"
fi


if [[ $1 == "clean" ]]; then
    for i in "${CUSRC[@]}"; do
        if [ -f "$OBJDIR/$i.o" ]; then
           rm $OBJDIR/$i.o
        fi
    done
fi
openMPFlags=""
#openMPFlags="-DopenMP --compiler-options -fopenmp"

#Compiling .cu files
for i in "${CUSRC[@]}"; do
    if [[ "${SRCDIR}/${i}.cu" -nt "${OBJDIR}/${i}.o" ]] || [[ "${SRCDIR}/${i}.cuh" -nt "${OBJDIR}/${i}.o" ]]; then
        echo Compiling $SRCDIR/$i
        nvcc -ccbin=$CC --compiler-options -Wall $INC $DIRS $openMPFlags -dc $ARCH $FLAGS -o $OBJDIR/${i}.o -c $SRCDIR/${i}.cu $LIBS 
    fi
done
for i in "${SRC[@]}"; do
    if [[ ! -f $OBJDIR/$i.o ]]; then
        echo Error compiling. Exiting program.
        exit 1
    fi
done
echo Creating Archives
LIBNAME="libgpuconvolve.a"
nvcc -ccbin g++ --lib $OBJDIR/Main.o $OBJDIR/FFT.o $OBJDIR/MemManage.o $OBJDIR/timeDomain.o \
    $OBJDIR/Convolution.o $OBJDIR/sgDevConvolution.o $OBJDIR/thrustOps.o  $OBJDIR/Audio.o\
    ${DIRS} ${LIBS} -o $LIBNAME
if [[ $1 == "noexec" ]]; then
    exit 0
else
    echo Creating executable
    nvcc -o bin/gpuconvolve.out $OBJDIR/runner.o $DIRS $LIBS -lgpuconvolve
fi