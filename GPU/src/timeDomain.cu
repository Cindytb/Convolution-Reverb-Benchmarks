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
float *TDconvolution(passable *p){
	float *d_ibuf = p->input->d_buf;
	float *d_rbuf = p->reverb->d_buf;
	long long rFrames = p->reverb->frames;
	long long iFrames = p->input->frames;
	long long oFrames = rFrames + iFrames - 1;
	enum flags flag = p->type;
	int iCh = p->input->channels;
	int rCh = p->reverb->channels;
	int oCh = flag == mono_mono ? 1 : 2;
	float minmax, minmax2;
	float *d_obuf, *obuf;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	cudaEventRecord(start);
	
	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * oCh * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * oCh * sizeof(float)));


	minmax = DExtrema(d_ibuf, iFrames * iCh);
	long long smallerFrames = rFrames < iFrames  ? rFrames  : iFrames;
	long long biggerFrames = rFrames >= iFrames  ? rFrames  : iFrames;
	int smallerCh = rFrames < iFrames  ? rCh  : iCh;
	int biggerCh = rFrames >= iFrames  ? rCh  : iCh;

	float *biggerBuf, *smallerBuf;
	if(biggerFrames == rFrames){
			biggerBuf = d_rbuf;
			smallerBuf = d_ibuf;
	}
	else{
		biggerBuf = d_ibuf;
		smallerBuf = d_rbuf;
	}
	int numThreads = 512;
	int numBlocks = (oFrames + numThreads - 1) / numThreads;

	cudaStream_t streams[2];
	
	
	if(flag == mono_mono){
		timeDomainConvolutionNaive<<<numBlocks, numThreads>>> (biggerBuf, smallerBuf, 
			d_obuf, biggerFrames, smallerFrames);
	}
	else if(flag == stereo_stereo){
		checkCudaErrors(cudaStreamCreate(&streams[0]));
		checkCudaErrors(cudaStreamCreate(&streams[1]));
		timeDomainConvolutionNaive<<<numBlocks, numThreads, 0, streams[0]>>> (biggerBuf, smallerBuf, 
			d_obuf, biggerFrames, smallerFrames);
		timeDomainConvolutionNaive<<<numBlocks, numThreads, 0, streams[1]>>> (biggerBuf + biggerFrames, 
			smallerBuf + smallerFrames, d_obuf + oFrames, biggerFrames, smallerFrames);
	}
	else{
		checkCudaErrors(cudaStreamCreate(&streams[0]));
		checkCudaErrors(cudaStreamCreate(&streams[1]));
		timeDomainConvolutionNaive<<<numBlocks, numThreads, 0, streams[0]>>> (biggerBuf, smallerBuf, d_obuf, 
			biggerFrames, smallerFrames);
		if(biggerCh > smallerCh){
			timeDomainConvolutionNaive<<<numBlocks, numThreads, 0, streams[1]>>> (biggerBuf + biggerFrames, 
				smallerBuf, d_obuf + oFrames, biggerFrames, smallerFrames);
		}
		else{
			timeDomainConvolutionNaive<<<numBlocks, numThreads, 0, streams[1]>>> (biggerBuf, 
				smallerBuf + smallerFrames, d_obuf + oFrames, biggerFrames, smallerFrames);
		}
	
	}
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	minmax2 = DExtrema(d_obuf, oFrames * oCh);
	float scale = minmax/minmax2;
	RealFloatScale <<< numBlocks, numThreads >>> (d_obuf, oFrames * oCh, scale);
	checkCudaErrors(cudaDeviceSynchronize());
	// Copy device memory to host
	checkCudaErrors(cudaMemcpy(obuf, d_obuf, oFrames * oCh * sizeof(float),
		cudaMemcpyDeviceToHost));
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);	
	checkCudaErrors(cudaFree(d_obuf));
	checkCudaErrors(cudaFree(d_ibuf));
	checkCudaErrors(cudaFree(d_rbuf));
	return obuf;
}