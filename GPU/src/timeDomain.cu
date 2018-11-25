#include "Convolution.cuh"
#define tile 512

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
float *TDconvolution(float ** d_ibuf, float ** d_rbuf, long long size, long long old_size, long long oFrames){
	float *d_obuf, *obuf;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * sizeof(float)));

	float minmax = DExtrema(*d_ibuf, old_size);
	long long rFrames = oFrames - old_size + 1;
	long long smallerFrames = rFrames < old_size  ? rFrames  : old_size;
	long long biggerFrames = rFrames >= old_size  ? rFrames  : old_size;
	
	float *biggerBuf, *smallerBuf;
	if(biggerFrames == rFrames){
			biggerBuf = *d_rbuf;
			smallerBuf = *d_ibuf;
	}
	else{
		biggerBuf = *d_ibuf;
		smallerBuf = *d_rbuf;
	}
	int numThreads = 512;
	int numBlocks = (oFrames + numThreads - 1) / numThreads;
	timeDomainConvolutionPlain<<<numBlocks, numThreads>>> (biggerBuf, smallerBuf, d_obuf, biggerFrames, smallerFrames);
	
	float minmax2 = DExtrema(d_obuf, oFrames);
	float scale = minmax/minmax2;
	RealFloatScale <<< numBlocks, numThreads >>> (d_obuf, oFrames, scale);
	checkCudaErrors(cudaDeviceSynchronize());
	// Copy device memory to host
	checkCudaErrors(cudaMemcpy(obuf, d_obuf, oFrames * sizeof(float),
		cudaMemcpyDeviceToHost));
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);	checkCudaErrors(cudaFree(d_obuf));
	checkCudaErrors(cudaFree(*d_ibuf));
	checkCudaErrors(cudaFree(*d_rbuf));
	return obuf;
}


/*RECYCLING BIN*/
/*

__global__ void timeDomainConvolution(float *ibuf, float *rbuf, float *obuf, long long iframes, long long rframes){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID < iframes + rframes - 1 && threadID > 32){
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

__global__ void timeDomainConvolutionStreamed(float *ibuf, float *rbuf, float *obuf, long long iframes, long long rframes, long long offset){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID < iframes + rframes - 1 && threadID > 32 && threadID < offset){
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
__global__ void preliminaryTimeDomainConvolution(float *ibuf, float *rbuf, float *obuf){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float X[tileSize];
	__shared__ float H[tileSize];
	float value = 0.0f;
	X[threadID] = ibuf[threadID];
	H[threadID] = rbuf[threadID];
	__syncthreads();
	for(int k = 0; k < tileSize; k++){
		if(threadID - k >= 0){
			break;
		}
		value += X[threadID - k] * H[k];
	}
	obuf[threadID] = value;
}
/*
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
*/
// float* timeDomainConvolutionExperimental(float ** d_ibuf, float ** d_rbuf, long long old_size, long long oFrames){
// 	cudaProfilerStart();
// 	float *d_obuf, *obuf;
// 	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));

// 	#if PINNED == 0
// 		obuf = (float*)malloc(oFrames * sizeof(float));
// 	#else
// 		checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * sizeof(float)));
// 	#endif
	
// 	//fprintf(stderr, "Finding max of input signal\n");
// 	float minmax = DExtrema(*d_ibuf, old_size);
// 	//printf("max of input: %f\n", minmax);
// 	fprintf(stderr, "Running Time Domain Convolution\n");
// 	long long rFrames = oFrames + 1 - old_size;
// 	long long smallerFrames = rFrames < old_size ? rFrames : old_size;
// 	long long biggerFrames = rFrames > old_size ? rFrames : old_size;
// 	float *d_biggerBuf;
// 	float *d_smallerBuf;
// 	if(biggerFrames == rFrames){
// 		d_biggerBuf = *d_rbuf;
// 		d_smallerBuf = *d_ibuf;
// 	}
// 	else{
// 		d_biggerBuf = *d_ibuf;
// 		d_smallerBuf = *d_rbuf;
// 	}
// 	int iFrames = old_size;
// 	fprintf(stderr, "Allocating a ton of memory for output testing\n");
// 	float *algo1 = (float*)malloc(oFrames * sizeof(float));
// 	float *algo2 = (float*)malloc(oFrames * sizeof(float));
// 	float *ibuf = (float*)malloc(iFrames * sizeof(float));
// 	float *rbuf = (float*)malloc(rFrames * sizeof(float));
// 	checkCudaErrors(cudaMemcpy(ibuf, *d_ibuf, iFrames * sizeof(float), cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaMemcpy(rbuf, *d_rbuf, rFrames * sizeof(float), cudaMemcpyDeviceToHost));
// 	cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);
// 	/*Setting all CPU outputs to 0*/
// 	for(int i = 0; i < oFrames; i++){
// 		algo1[i] = 0.0f;
// 		algo2[i] = 0.0f;
// 	}
// 	cudaEventRecord(start);
// 	fprintf(stderr, "Performing algorithm 1 on CPU\n");
// 	for(int i = 0; i < oFrames; i++){
// 		for (int k = 0; k < rFrames; k++){
// 			if(i - k >=0 && i - k < iFrames){
// 				algo1[i] += ibuf[i - k] * rbuf[k];
// 			}
// 		}
// 	}
// 	cudaEventRecord(stop);
// 	float milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "Time for Algorithm 1: %f ms\n", milliseconds);

// 	cudaEventRecord(start);
// 	fprintf(stderr, "Performing algorithm 2 on CPU\n");
// 	for(int k = 0; k < rFrames; k++){
// 		for(int n = 0; n < iFrames; n++){
// 			algo2[k + n] += ibuf[n] * rbuf[k];
// 		}
// 	}
// 	cudaEventRecord(stop);
// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "Time for Algorithm 2: %f ms\n\n", milliseconds);

// 	fprintf(stderr, "Error Check Algorithm 1 vs Algorithm 2 %s\n", errorCheckBufs(algo1, algo2, oFrames) == 0 ? "SUCCESS" : "FAILURE");
// 	float *d_atomic, *d_atomic_shared, *d_naive, *d_excessive;

// 	checkCudaErrors(cudaMalloc(&d_atomic, oFrames * sizeof(float)));
// 	checkCudaErrors(cudaMalloc(&d_atomic_shared, oFrames * sizeof(float)));
// 	checkCudaErrors(cudaMalloc(&d_naive, oFrames * sizeof(float)));
// 	checkCudaErrors(cudaMalloc(&d_excessive, oFrames * sizeof(float)));
	
// 	/*Filling all test buffers to zero*/
// 	int numThreads = 1024;
// 	int numBlocks = (oFrames + numThreads - 1) / numThreads;
// 	FillWithZeros<<<numBlocks, numThreads>>>(d_atomic, 0.0f, oFrames);
// 	FillWithZeros<<<numBlocks, numThreads>>>(d_atomic_shared, 0.0f, oFrames);
// 	FillWithZeros<<<numBlocks, numThreads>>>(d_naive, 0.0f, oFrames);
// 	FillWithZeros<<<numBlocks, numThreads>>>(d_excessive, 0.0f, oFrames);
	


// 	/*ATOMIC*/
// 	cudaEventRecord(start);
// 	fprintf(stderr, "\n\nLaunching Atomic\n\n");
// 	numBlocks = (smallerFrames + numThreads - 1) / numThreads;
// 	timeDomainConvolutionAtomic<<< numBlocks, numThreads >>> (d_biggerBuf, d_smallerBuf, d_atomic, biggerFrames, smallerFrames);
// 	cudaEventRecord(stop);
// 	cudaEventSynchronize(stop);
// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "Time for timeDomainConvolutionAtomic: %f ms\n", milliseconds);


// 	checkCudaErrors(cudaMemcpy(obuf, d_atomic, oFrames * sizeof(float), cudaMemcpyDeviceToHost));
// 	fprintf(stderr, "\n\nError Check Atomic vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
// 	fprintf(stderr, "\n\nError Check Atomic vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
	
	
// 	/*ATOMIC SHARED*/
// 	numBlocks = (oFrames + numThreads - 1) / numThreads;
// 	cudaEventRecord(start);
// 	numBlocks = (smallerFrames + tile - 1) / tile;
// 	timeDomainConvolutionAtomicShared<<<numBlocks, tile >>>(d_biggerBuf, d_smallerBuf, d_atomic_shared, biggerFrames, smallerFrames);
// 	cudaEventRecord(stop);
// 	cudaEventSynchronize(stop);
// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "\n\nTime for Atomic Shared: %f ms\n\n", milliseconds);

// 	checkCudaErrors(cudaMemcpy(obuf, d_atomic_shared, oFrames * sizeof(float),
// 		cudaMemcpyDeviceToHost));
		
// 	fprintf(stderr, "\n\nError Check Atomic Shared vs Algorithm 1 %s\n\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
// 	fprintf(stderr, "\n\nError Check Atomic Shared vs Algorithm 2 %s\n\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

// 	/*NAIVE*/
// 	cudaEventRecord(start);
// 	numBlocks = (oFrames + numThreads - 1) / numThreads;
// 	fprintf(stderr, "\n\nLaunching Naive\n\n");
// 	timeDomainConvolutionPlain<<<numBlocks, numThreads >>>(d_biggerBuf, d_smallerBuf, d_naive, biggerFrames, smallerFrames);
// 	cudaEventRecord(stop);
// 	cudaEventSynchronize(stop);
// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "\n\nTime for Naive: %f ms\n\n", milliseconds);

// 	checkCudaErrors(cudaMemcpy(obuf, d_naive, oFrames * sizeof(float),
// 		cudaMemcpyDeviceToHost));
	
// 	fprintf(stderr, "\n\nError Check Naive vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
// 	fprintf(stderr, "\n\nError Check Naive vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

// 	/*EXCESSIVE*/
// 	fprintf(stderr, "Launching Excessive\n");
// 	cudaEventRecord(start);
// 	int numChunks = (biggerFrames + tile - 1) / tile;
// 	cudaStream_t streams[numChunks];
// 	for(int i = 0; i < numChunks; i++){
// 		checkCudaErrors(cudaStreamCreate(&streams[i]));
// 		timeDomainConvolutionExcessive<<<iFrames, tile, 0, streams[i]>>>(d_biggerBuf, d_smallerBuf, d_excessive, iFrames, i);
// 	}
// 	checkCudaErrors(cudaDeviceSynchronize());
// 	for(int i = 0; i < numChunks; i++){
// 		checkCudaErrors(cudaStreamDestroy(streams[i]));
// 	}
// 	cudaEventRecord(stop);
// 	cudaEventSynchronize(stop);
// 	milliseconds = 0;
// 	cudaEventElapsedTime(&milliseconds, start, stop);
// 	fprintf(stderr, "\n\nTime for Excessive: %f ms\n\n", milliseconds);
// 	checkCudaErrors(cudaMemcpy(obuf, d_excessive, oFrames * sizeof(float),
// 	cudaMemcpyDeviceToHost));
// 	fprintf(stderr, "\n\nError Check Excessive vs Algorithm 1 %s\n", errorCheckBufs(algo1, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");
// 	fprintf(stderr, "\n\nError Check Excessive vs Algorithm 2 %s\n", errorCheckBufs(algo2, obuf, oFrames) == 0 ? "SUCCESS" : "FAILURE");

	
// 	cudaFree(d_atomic);
// 	cudaFree(d_atomic_shared);
// 	cudaFree(d_naive);
// 	cudaFree(d_excessive);
// 	free(algo1);
// 	free(algo2);
// 	free(ibuf);
//     free(rbuf);



//     /*Remainder for the context*/
//     float minmax2 = DExtrema(d_obuf, oFrames);
// 	float scale = minmax/minmax2;
// 	int strides = 1;
// 	int blockSize = 128;
// 	numBlocks = (oFrames / strides + blockSize - 1) / blockSize;
// 	while (numBlocks > (2U << 31 - 1)) {
// 		numBlocks = (oFrames / ++strides + blockSize - 1) / blockSize;
// 	}
	
// 	RealFloatScale <<< numBlocks, blockSize >>> (d_obuf, oFrames, scale);
// 	checkCudaErrors(cudaMemcpy(obuf, d_obuf, oFrames * sizeof(float),
// 		cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaFree(d_obuf));
// 	cudaProfilerStop();
// 	return obuf;

// }
