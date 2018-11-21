#include "Convolution.cuh"
#define tile 512

/*
Utility functions defined elsewhere in program:

__global__ void FillWithZeros(float *buf, long long start, long long size) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID + start < size)
		buf[threadID + start] = 0.0f;
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

#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
bool check(T result, char const *const func, const char *const file, int const line){
    if (result){
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
*/
__global__ void timeDomainConvolutionAtomic(float *ibuf, float *rbuf, float *obuf, long long iFrames, long long rFrames){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	float h;
	if(threadID < rFrames){
		h = rbuf[threadID];
		for(int i = 0; i < iFrames; i++){
			atomicAdd(obuf + threadID  + i , ibuf[i] * h);
		}
	}
	
}

__global__ void timeDomainConvolutionAtomicShared(float *ibuf, float *rbuf, float *obuf, long long iFrames, long long rFrames){
	__shared__ float x[tile];
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int numLoops = (iFrames + tile - 1) / tile;
	float h = 0.0f;
	if(threadID < rFrames){
		h = rbuf[threadID];
	}
	for(int n = 0; n < numLoops; n++){
		if(n * tile + threadIdx.x < iFrames){
			x[threadIdx.x] = ibuf[n * tile + threadIdx.x];
		}
		__syncthreads();
		for(int i = 0; i < tile; i++){
			if(n * tile + i < iFrames){
				atomicAdd(obuf + threadID + (n * tile) + i , x[i] * h);
			}
		}
		__syncthreads();
	}

}

__global__ void timeDomainConvolutionExcessive(float *ibuf, float *rbuf, float *obuf, long long iFrames, int chunkNo){
	__shared__ float x[tile];
	__shared__ float h;
	
	if(chunkNo * tile + threadIdx.x < iFrames){
		x[threadIdx.x] = ibuf[chunkNo * tile + threadIdx.x];
	}
	h = rbuf[blockIdx.x];
		__syncthreads();
	if(chunkNo * tile + threadIdx.x < iFrames){
		atomicAdd(obuf + blockIdx.x + chunkNo * tile + threadIdx.x, x[threadIdx.x] * h);
	}
}

__global__ void timeDomainConvolutionPlain(float *ibuf, float *rbuf, float *obuf, long long iframes, long long rframes){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID < iframes + rframes - 1){
		int i_size = iframes;
		float value = 0;
		for(int k = 0; k < rframes; k++){
			if(threadID - k >= 0 && threadID - k <= i_size){
				value += ibuf[threadID - k] * rbuf[k];
			}
		}
		obuf[threadID] = value;
	}
}
float* timeDomainConvolutionExperimental(float ** d_ibuf, float ** d_rbuf, long long old_size, long long oFrames){
	/*Necessary for the context of the function*/
	float *d_obuf, *obuf;
	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * sizeof(float)));
	float minmax = DExtrema(*d_ibuf, old_size);

    /*TEST SECTION----------------------------------------------------------------------------------------------------------------------------*/
    fprintf(stderr, "Running Time Domain Convolution\n");
    
    /*swap input/reverb depending on which is smaller*/
    /*reverb refers to the filter kernel*/
	long long rFrames = oFrames + 1 - old_size;
	long long smallerFrames = rFrames < old_size ? rFrames : old_size;
	long long biggerFrames = rFrames > old_size ? rFrames : old_size;
	float *d_biggerBuf;
	float *d_smallerBuf;
	if(biggerFrames == rFrames){
		d_biggerBuf = *d_rbuf;
		d_smallerBuf = *d_ibuf;
	}
	else{
		d_biggerBuf = *d_ibuf;
		d_smallerBuf = *d_rbuf;
	}
	int iFrames = old_size;
	fprintf(stderr, "Allocating a ton of memory for output testing\n");
	float *algo1 = (float*)malloc(oFrames * sizeof(float));
	float *algo2 = (float*)malloc(oFrames * sizeof(float));
	float *ibuf = (float*)malloc(iFrames * sizeof(float));
	float *rbuf = (float*)malloc(rFrames * sizeof(float));
	checkCudaErrors(cudaMemcpy(ibuf, *d_ibuf, iFrames * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rbuf, *d_rbuf, rFrames * sizeof(float), cudaMemcpyDeviceToHost));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
    /*Setting all CPU test buffers to 0*/
	for(int i = 0; i < oFrames; i++){
		algo1[i] = 0.0f;
		algo2[i] = 0.0f;
	}
	cudaEventRecord(start);
	fprintf(stderr, "Performing algorithm 1 on CPU\n");
	for(int i = 0; i < oFrames; i++){
		for (int k = 0; k < rFrames; k++){
			if(i - k >=0 && i - k < iFrames){
				algo1[i] += ibuf[i - k] * rbuf[k];
			}
		}
	}
	cudaEventRecord(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for Algorithm 1: %f ms\n", milliseconds);

	cudaEventRecord(start);
	fprintf(stderr, "Performing algorithm 2 on CPU\n");
	for(int k = 0; k < rFrames; k++){
		for(int n = 0; n < iFrames; n++){
			algo2[k + n] += ibuf[n] * rbuf[k];
		}
	}
	cudaEventRecord(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for Algorithm 2: %f ms\n\n", milliseconds);

	fprintf(stderr, "Error Check Algorithm 1 vs Algorithm 2 %s\n", errorCheckBufs(algo1, algo2, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	float *d_atomic, *d_atomic_shared, *d_naive, *d_excessive;

	checkCudaErrors(cudaMalloc(&d_atomic, oFrames * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_atomic_shared, oFrames * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_naive, oFrames * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_excessive, oFrames * sizeof(float)));
	
	/*Filling all GPU test buffers to zero*/
	int numThreads = 1024;
	int numBlocks = (oFrames + numThreads - 1) / numThreads;
	FillWithZeros<<<numBlocks, numThreads>>>(d_atomic, 0.0f, oFrames);
	FillWithZeros<<<numBlocks, numThreads>>>(d_atomic_shared, 0.0f, oFrames);
	FillWithZeros<<<numBlocks, numThreads>>>(d_naive, 0.0f, oFrames);
	FillWithZeros<<<numBlocks, numThreads>>>(d_excessive, 0.0f, oFrames);
	


	/*ATOMIC*/
	cudaEventRecord(start);
	fprintf(stderr, "\n\nLaunching Atomic\n\n");
	numBlocks = (smallerFrames + numThreads - 1) / numThreads;
	timeDomainConvolutionAtomic<<< numBlocks, numThreads >>> (d_biggerBuf, d_smallerBuf, d_atomic, biggerFrames, smallerFrames);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for timeDomainConvolutionAtomic: %f ms\n", milliseconds);


	checkCudaErrors(cudaMemcpy(obuf, d_atomic, oFrames * sizeof(float), cudaMemcpyDeviceToHost));
	fprintf(stderr, "\n\nError Check Atomic vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	fprintf(stderr, "\n\nError Check Atomic vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	
	
	/*ATOMIC SHARED*/
	numBlocks = (oFrames + numThreads - 1) / numThreads;
	cudaEventRecord(start);
	numBlocks = (smallerFrames + tile - 1) / tile;
	timeDomainConvolutionAtomicShared<<<numBlocks, tile >>>(d_biggerBuf, d_smallerBuf, d_atomic_shared, biggerFrames, smallerFrames);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "\n\nTime for Atomic Shared: %f ms\n\n", milliseconds);

	checkCudaErrors(cudaMemcpy(obuf, d_atomic_shared, oFrames * sizeof(float),
		cudaMemcpyDeviceToHost));
		
	fprintf(stderr, "\n\nError Check Atomic Shared vs Algorithm 1 %s\n\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	fprintf(stderr, "\n\nError Check Atomic Shared vs Algorithm 2 %s\n\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

	/*NAIVE*/
	cudaEventRecord(start);
	numBlocks = (oFrames + numThreads - 1) / numThreads;
	fprintf(stderr, "\n\nLaunching Naive\n\n");
	timeDomainConvolutionPlain<<<numBlocks, numThreads >>>(d_biggerBuf, d_smallerBuf, d_naive, biggerFrames, smallerFrames);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "\n\nTime for Naive: %f ms\n\n", milliseconds);

	checkCudaErrors(cudaMemcpy(obuf, d_naive, oFrames * sizeof(float),
		cudaMemcpyDeviceToHost));
	
	fprintf(stderr, "\n\nError Check Naive vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	fprintf(stderr, "\n\nError Check Naive vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

	/*EXCESSIVE*/
	fprintf(stderr, "Launching Excessive\n");
	cudaEventRecord(start);
	int numChunks = (biggerFrames + tile - 1) / tile;
	cudaStream_t streams[numChunks];
	for(int i = 0; i < numChunks; i++){
		checkCudaErrors(cudaStreamCreate(&streams[i]));
		timeDomainConvolutionExcessive<<<iFrames, tile, 0, streams[i]>>>(d_biggerBuf, d_smallerBuf, d_excessive, iFrames, i);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	for(int i = 0; i < numChunks; i++){
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "\n\nTime for Excessive: %f ms\n\n", milliseconds);
	checkCudaErrors(cudaMemcpy(obuf, d_excessive, oFrames * sizeof(float),
	cudaMemcpyDeviceToHost));
	fprintf(stderr, "\n\nError Check Excessive vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	fprintf(stderr, "\n\nError Check Excessive vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

	/*Free memory*/
	cudaFree(d_atomic);
	cudaFree(d_atomic_shared);
	cudaFree(d_naive);
	cudaFree(d_excessive);
	free(algo1);
	free(algo2);
	free(ibuf);
    free(rbuf);


    /*------------------------------------------------------------------------------------------------------------------------------------*/
    /*Remainder code is for the context of the function*/
    float minmax2 = DExtrema(d_obuf, oFrames);
	float scale = minmax/minmax2;
	int strides = 1;
	int blockSize = 128;
	numBlocks = (oFrames / strides + blockSize - 1) / blockSize;
	while (numBlocks > (2U << 31 - 1)) {
		numBlocks = (oFrames / ++strides + blockSize - 1) / blockSize;
	}
	
	RealFloatScale <<< numBlocks, blockSize >>> (d_obuf, oFrames, scale);
	checkCudaErrors(cudaMemcpy(obuf, d_obuf, oFrames * sizeof(float),
		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_obuf));
	cudaProfilerStop();
	return obuf;

}