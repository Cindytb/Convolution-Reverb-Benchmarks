#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include "fftwconvolve.h"
#include "Audio.h"

void TDconvolution(float *ibuf, float *rbuf, size_t iframes, size_t rframes, int iCh, int rCh, float *obuf);
void maxScale(float *ibuf, long long iframes, long long oSize, float *obuf);
float* seqEntry(std::string input, std::string reverb, std::string out, bool timeDomain);
