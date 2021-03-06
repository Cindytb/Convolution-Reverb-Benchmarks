#include <math.h>
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <locale.h>

#ifndef __INT_MAX__
#define __INT_MAX__ INT_MAX
#endif
float *blockConvolve(float *ibuf, float *rbuf, long long iFrames, long long rFrames, int iCh, int rCh);
float * regularConvolve(float *ibuf, long long paddedSize, long long iFrames, long long oFrames, int iCh, int rCh);