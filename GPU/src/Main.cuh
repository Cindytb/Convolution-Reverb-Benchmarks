
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include "Convolution.cuh"
#include "MemManage.cuh"
#include "Audio.cuh"


float *gpuEntry(std::string input, std::string reverb, std::string out, bool timeDomain);

