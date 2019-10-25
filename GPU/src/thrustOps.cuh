/*Thrust includes*/
#include "Universal.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <cmath>

#include <helper_functions.h>
#include <helper_cuda.h>

void fillWithZeroes(float **target_buf, long long old_size, long long new_size);
float DExtrema(float *pointer, long long size);