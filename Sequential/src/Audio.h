#pragma once

#include <stdlib.h>
#include <string.h>
#include <sndfile.h>
#include <sndfile.hh>
#include <math.h>
#include <fftw3.h>
#define MAX_CHN 2

long long readFile(const char *name, float **buf, int *numCh, int *SR);
long long readFileStereo(const char *name, float **buf, int *numCh, int *SR);

void readFileExperimental(const char *iname, const char *rname, 
	int *iCh, int *iSR, long long *iframes, int *rCh, int *rSR,  long long *rframes, 
	float **ibuf, float **rbuf, long long *new_size, bool *blockProcessingOn, bool timeDomain);
void cpuWriteFile(const char * name, float * buf, long long size, int fs, int ch);
