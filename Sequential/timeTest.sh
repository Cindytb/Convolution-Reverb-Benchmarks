#!/bin/bash
module load gcc/6.3.0
module load libsndfile/intel/1.0.28
module load fftw/intel/3.3.6-pl2

(time bin/seqConvolve.out -t -i $1 )  2>&1