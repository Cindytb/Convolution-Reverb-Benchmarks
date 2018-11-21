// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "thrustOps.cuh"
#include "Universal.h"
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

int errorCheckBufs(float *buf1, float *buf2, size_t size);
typedef float2 Complex;

// Complex pointwise multiplication
__global__ void ComplexPointwiseMul(Complex *a, const Complex *b, int size);

// This is the callback routine. It does complex pointwise multiplication with scaling.
__device__ cufftComplex cbComplexPointwiseMul(void *dataIn, size_t offset, void *cb_info, void *sharedmem); 

//Scaling real arrays
__global__ void RealFloatScale(float *a, long long size, float scale);

//Scaling real arrays w/ diff streams
__global__ void RealFloatScaleConcurrent(float *a, long long size, long long streamSize, float scale, int offset);

__global__ void PointwiseAdd(float *a, float *b, int size);
