#!/bin/bash
#SBATCH --gres=gpu:1
host=$(hostname)
if [[ "$host" = *"cims"* ]]
    then host="cims"
    else host="prince"
fi


LIBS="-lm -lcufft_static -lculibos -lsndfile -lnvToolsExt "
CSRC=()
CUSRC=("Main" "Convolution" "MemManage" "FFT" "Audio" "timeDomain" "runner")
#Not compiling ThrustOps or runner because it never seems to change
#FLAGS="-Xptxas=-v"
BINDIR="bin"
OBJDIR="obj"
SRCDIR="src"
mkdir -p $BINDIR
mkdir -p $OBJDIR

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
    
else
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
fi


if [[ $1 == "clean" ]]; then
    for i in "$OBJDIR/${CUSRC[@]}"; do
        if [ -f "$OBJDIR/$i.o" ]; then
           rm $OBJDIR/$i.o
        fi
    done
fi



#Compiling .cu files
for i in "${CUSRC[@]}"; do
    if [[ "${SRCDIR}/${i}.cu" -nt "${OBJDIR}/${i}.o" ]] || [[ "${SRCDIR}/${i}.cuh" -nt "${OBJDIR}/${i}.o" ]]; then
        echo Compiling $SRCDIR/$i
        nvcc $INC $DIRS -dc $ARCH $FLAGS \
            -o $OBJDIR/${i}.o -c $SRCDIR/${i}.cu \
            $LIBS 
            
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
nvcc -ccbin g++ --lib $OBJDIR/FFT.o $OBJDIR/MemManage.o $OBJDIR/timeDomain.o \
    $OBJDIR/Convolution.o  $OBJDIR/thrustOps.o $OBJDIR/Main.o $OBJDIR/Audio.o\
    ${DIRS} ${LIBS} -o $LIBNAME
if [[ $1 == "noexec" ]]; then
    exit 0
else
    echo Creating executable
    nvcc -o bin/gpuconvolve.out obj/runner.o $DIRS $LIBS -lgpuconvolve
fi