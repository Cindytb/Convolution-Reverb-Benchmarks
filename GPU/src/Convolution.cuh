#pragma once
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <locale.h>

#include "thrustOps.cuh"
#include "FFT.cuh"
#include "MemManage.cuh"
#include "Audio.cuh"
#include "Universal.h"
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>

float * TDconvolution(float ** d_ibuf, float ** d_rbuf, long long old_size, long long written_size);
float * convolution(float ** d_ibuf, float ** d_rbuf, long long new_size, long long old_size, long long written_size);
float * blockConvolution(float ** d_ibuf, float ** d_rbuf, long long old_size, long long written_size, long long audioBlockSize);
//float * timeDomainConvolutionExperimental(float ** d_ibuf, float ** d_rbuf, long long old_size, long long written_size);
float *multiGPUFFT(float *ibuf, float *rbuf, long long iFrames, long long rFrames);

// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#ifndef CHECK_CUFFT_ERRORS
#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}
#endif