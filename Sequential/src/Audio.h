#pragma once

#include <stdlib.h>
#include <string.h>
#include <sndfile.h>
#include <sndfile.hh>
#define MAX_CHN 2

long long readFile(const char *name, float **buf, int *numCh, int *SR);
long long readFileStereo(const char *name, float **buf, int *numCh, int *SR);
void cpuWriteFile(const char * name, float * buf, long long size, int fs, int ch);
