#include "FFT.cuh"

// Complex multiplication
 __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

// Complex pointwise multiplication
__global__ void ComplexPointwiseMul(Complex *a, const Complex *b, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = ComplexMul(a[i], b[i]);
	}
}
// This is the callback routine. It does complex pointwise multiplication with scaling.
__device__ cufftComplex cbComplexPointwiseMul(void *dataIn, size_t offset, void *cb_info, void *sharedmem) {
	cufftComplex *filter = (cufftComplex*)cb_info;
	return (cufftComplex) ComplexMul(((Complex *)dataIn)[offset], filter[offset]);
}


__global__ void PointwiseAdd(float *a, float *b, int size) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < size) {
		b[threadID] += a[threadID];
	}
}


//Scaling real arrays
__global__ void RealFloatScale(float *a, long long size, float scale) {
	int numThreads = blockDim.x * gridDim.x;
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (; threadID < size; threadID += numThreads) {
		a[threadID] *= scale;
	}
}

//Scaling real arrays w/ diff streams
__global__ void RealFloatScaleConcurrent(float *a, long long size, long long streamSize, float scale, int offset) {
	//const int numThreads = blockDim.x * gridDim.x;
	const int threadID = offset + blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < size && threadID - offset < streamSize) {
		a[threadID] *= scale;
	}
}

int errorCheckBufs(float *buf1, float *buf2, size_t size){
	float *buf3 = (float *)malloc(size * sizeof(float));
    int max = 0;
    for(long long i = 0; i < size; i++){
        buf3[i] = fabs(buf1[i] - buf2[i]);
        if (buf3[max] < fabs(buf3[i])){
            max = i;
        }
	}
	float epsilon = 1e-6f;
    fprintf(stderr,"\nEpsilon for this program is %e\n", epsilon);
	fprintf(stderr,"The maximum difference between the two buffers is at sample %i\n",  max);
    fprintf(stderr,"buf1[max] = %11.8f\n", buf1[max]);
    fprintf(stderr,"buf2[max] = %11.8f\n", buf2[max]);
	fprintf(stderr,"Difference= %E\n", buf3[max]);
	int returnval = buf3[max] < epsilon ? 0 : 1;
	free(buf3);
	return returnval;
}