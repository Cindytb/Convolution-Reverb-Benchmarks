#pragma once

#include "Universal.h"
#include "thrustOps.cuh"

// includes, project
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
__global__ void FillWithZeros(float *buf, long long start, long long size);
void printSize();
size_t getFreeSize();
