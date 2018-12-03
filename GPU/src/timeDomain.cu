#include "Convolution.cuh"

__global__ void timeDomainConvolutionNaive(float *ibuf, float *rbuf, float *obuf, long long iframes, long long rframes){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID < iframes + rframes - 1){
		float value = 0;
		for(int k = 0; k < rframes; k++){
			if(threadID - k >= 0 && threadID - k <= iframes){
				value += ibuf[threadID - k] * rbuf[k];
			}
		}
		obuf[threadID] = value;
	}
}
float *TDconvolution(float ** d_ibuf, float ** d_rbuf, long long iFrames, long long oFrames){
	float *d_obuf, *obuf;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * sizeof(float)));

	float minmax = DExtrema(*d_ibuf, iFrames);
	long long rFrames = oFrames - iFrames + 1;
	long long smallerFrames = rFrames < iFrames  ? rFrames  : iFrames;
	long long biggerFrames = rFrames >= iFrames  ? rFrames  : iFrames;
	
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
	timeDomainConvolutionNaive<<<numBlocks, numThreads>>> (biggerBuf, smallerBuf, d_obuf, biggerFrames, smallerFrames);
	checkCudaErrors(cudaDeviceSynchronize());
	
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
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);	
	checkCudaErrors(cudaFree(d_obuf));
	checkCudaErrors(cudaFree(*d_ibuf));
	checkCudaErrors(cudaFree(*d_rbuf));
	return obuf;
}