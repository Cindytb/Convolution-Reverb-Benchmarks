#!/bin/bash
host=$(hostname)
if [[ "$host" = *"cims"* ]]
    then host="cims"
    else host="prince"
fi
DIRS="
-L/home/ctb335/libsndfile/lib
-L/share/apps/libsndfile/1.0.28/intel/lib \
-L/share/apps/fftw/3.3.6-pl2/intel/lib
-L/home/ctb335/Capstone/Sequential
-L/media/removable/CTB/Capstone/MASTER/Sequential
"

INC="
-I/home/ctb335/libsndfile/include \
-I/share/apps/libsndfile/1.0.28/intel/include \
-I/share/apps/fftw/3.3.6-pl2/intel/include
"

LIBS="-lm -lsndfile -lfftw3f"

BINDIR="bin"
SRC=("fftwconvolve" "main" "Audio")
OUT=("freqDomain.out" "timeDomain.out")
GCC="g++"
OBJDIR="obj"
SRCDIR="src"
if [[ $host = "cims" ]]; then
    module purge
    module load gcc-6.2.0
else
    module purge
    module load gcc/6.3.0
    module load libsndfile/intel/1.0.28
    module load fftw/intel/3.3.6-pl2
fi
if [[ $1 == "clean" ]]; then
    rm ${OBJDIR}/*.o
fi
for i in "${SRC[@]}"; do
    if [[ "${SRCDIR}/${i}.cpp" -nt "${OBJDIR}/${i}.o" ]]; then
        if [[ -e "${SRCDIR}/${i}.h"  ]] && [[ "${SRCDIR}/${i}.h" -nt "${OBJDIR}/${i}.o" ]]; then
            echo Compiling $SRCDIR/$i
            $GCC -fPIC -o "${OBJDIR}/${i}.o" -c "${SRCDIR}/${i}.cpp" $INC $DIRS ${DEFINES[round]}
        else
            echo Compiling $SRCDIR/$i
            $GCC -fPIC -o "${OBJDIR}/${i}.o" -c "${SRCDIR}/${i}.cpp" $INC $DIRS ${DEFINES[round]}
        fi
    fi
done
for i in "${SRC[@]}"; do
    if [[ ! -f $OBJDIR/$i.o ]]; then
        echo Error compiling. $i.o not found. Exiting program.
        exit 1
    fi
done
echo Creating Archives
ar rcs libseqconvolve.a obj/main.o obj/Audio.o obj/fftwconvolve.o 
if [[ $1 == "noexec" ]]; then
    exit 0
else
    echo Creating executable
    g++ -o bin/seqConvolve.out src/runner.cpp $INC $DIRS $LIBS -lseqconvolve
fi
