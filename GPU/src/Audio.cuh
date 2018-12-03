#pragma once

#include <locale.h>
#include <cmath>
#include <string.h>
#include <sndfile.hh>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include "MemManage.cuh"

#include "nvToolsExt.h"

void errorCheck(int iCh, int iSR, int rSR);

long long getAudioBlockSize();
void readFile(const char *iname, const char *rname, 
	int *iCh, int *iSR, long long *iframes, int *rCh, int *rSR,  long long *rframes, 
	float **d_ibuf, float **d_rbuf, long long *new_size, bool *blockProcessingOn, bool timeDomain);
void writeFile(const char * name, float * buf, long long size, int fs, int ch);
