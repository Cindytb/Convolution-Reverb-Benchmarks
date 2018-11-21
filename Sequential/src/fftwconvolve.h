#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <locale.h>

float *entry(float *ibuf, float *rbuf, long long iFrames, long long rFrames, int iCh, int rCh);