#pragma once

#include <locale.h>
#include <cmath>
#include <string.h>
#include <stdio.h> 
#include <math.h> 
#include <sndfile.hh>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include "MemManage.cuh"

#include "Universal.h"
#include "Main.cuh"

void printArr(float *buf, int len);
void errorCheckGPU(int iCh, int rCh, int iSR, int rSR, passable *p);
__host__ __device__ void deInterleave(float *buf, long long samples);
void interleave(float *buf, long long frames);
long long getAudioBlockSize(passable *p);
void readFileExperimentalDebug(const char *iname, const char *rname,
    int *SR, bool *blockProcessingOn, bool timeDomain, passable *p);
void readFileExperimental(const char *iname, const char *rname,
    int *SR, bool *blockProcessingOn, bool timeDomain, passable *p);
void writeFile(const char * name, float * buf, long long size, int fs, int ch);
