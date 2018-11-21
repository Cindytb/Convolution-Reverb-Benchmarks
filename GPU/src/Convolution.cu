#include "Convolution.cuh"
// Define the device pointer to the callback routine. The host code will fetch this and pass it to CUFFT
__device__ cufftCallbackLoadC myOwnCallbackPtr = cbComplexPointwiseMul;

void convolve(float **d_ibuf, float **d_rbuf, cufftComplex **d_Cbufs, long long size){
	cufftComplex *d_sig_complex = *d_Cbufs, *d_filter_complex = *d_Cbufs + size / 2 + 1;
	
	/*Create forward FFT plan*/
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, size, CUFFT_R2C, 1));

	/*Create inverse FFT plan*/
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, size, CUFFT_C2R, 1));

#if defined WIN64 || CALLBACK == 0
	/*NO CALLBACK VERSION*/

	/*Transform Complex Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *) *d_ibuf, d_sig_complex));

	/*Transform Filter Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*) *d_rbuf, d_filter_complex));
	checkCudaErrors(cudaFree(*d_rbuf));

	/*CONVOLUTION*/
	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	
	ComplexPointwiseMul << < numBlocks, blockSize >> > (d_sig_complex, d_filter_complex, size / 2 + 1);
	getLastCudaError("Kernel execution failed [ ComplexPointwiseMul]");
	
	/*IFFT*/
	CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, d_sig_complex, *d_ibuf));
#else
	/*Transform Complex Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)*d_ibuf, d_sig_complex));

	/*Transform Filter Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)*d_rbuf, d_filter_complex));
	
	/*Copy over the host copy of callback function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr,myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));
	
	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX, 
		(void **)&d_filter_complex));
	
	checkCudaErrors(cudaFree(*d_rbuf));

	// Transform signal back, using the callback to do the pointwise multiply on the way in.
	CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, d_sig_complex, *d_ibuf));
#endif
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(outplan));
	
	checkCudaErrors(cudaFree(d_sig_complex));
}

void blockConvolve(float **d_ibuf, float **d_rbuf, long long iFrames, long long rFrames){
	cufftComplex *d_sig_complex, *d_filter_complex;
	float *d_padded_signal;
	float *d_padded_filter_kernel;
	float *d_obuf = *d_ibuf;

	int M = rFrames - 1;
	
	int myExp = ceil(log2( (float)(iFrames +M)));
	size_t blockSize = pow(2, myExp);
	int L = iFrames;
	int blockNum = 0;
	size_t workspace;
	CHECK_CUFFT_ERRORS(cufftEstimate1d(blockSize, CUFFT_R2C, 2, &workspace));
	while(getFreeSize() < workspace + blockSize * 18L){
		myExp--;
		blockSize = pow(2, myExp);
		blockNum++;
		CHECK_CUFFT_ERRORS(cufftEstimate1d(blockSize, CUFFT_R2C, 2, &workspace));
	}
	if(blockSize < iFrames + M) L = blockSize - M;
	
	/*Allocating Memory*/
	checkCudaErrors(cudaMalloc(&d_filter_complex, (blockSize + 2) * sizeof(cufftComplex)));
	checkCudaErrors(cudaMalloc(&d_padded_filter_kernel, blockSize * sizeof(float)));
	d_sig_complex = d_filter_complex + blockSize / 2 + 1;	
	checkCudaErrors(cudaMalloc(&d_padded_signal, blockSize * sizeof(float)));

	/*Block/Thread sizes for kernels*/
	int numThreads = 256;
	int numBlocks = (blockSize + numThreads - 1) / numThreads;
	
	/* Copy over filter */
	checkCudaErrors(cudaMemcpy(d_padded_filter_kernel, *d_rbuf, rFrames * sizeof(float), cudaMemcpyDeviceToDevice));
	numBlocks = (rFrames + numThreads - 1) / numThreads;
	FillWithZeros<<<numBlocks, numThreads>>>(d_padded_filter_kernel, rFrames,  blockSize);
	
	/*Free real array*/
	checkCudaErrors(cudaFree(*d_rbuf));

	/*Plans*/
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, blockSize, CUFFT_R2C, 1));
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, blockSize, CUFFT_C2R, 1));

	/*Transform Filter*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_padded_filter_kernel, d_filter_complex));
	
	/*Free real padded array*/
	checkCudaErrors(cudaFree(d_padded_filter_kernel));
	
	
		
	#if defined WIN64 || CALLBACK == 0
	#else
	//fprintf(stderr, "DOING CALLBACK STUFF\n");
	/*Create host pointer to CB Function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr,myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));
		
	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr,
		 CUFFT_CB_LD_COMPLEX, (void **)&d_filter_complex));
	#endif	
	for(int blockNo = 0; blockNo <= blockNum; blockNo++){
		long long cpyAmount = L;
		if (blockNo == blockNum) {
			cpyAmount = iFrames % L;
		}
		//fprintf(stderr, "blockNo: %'i\tcpyAmount: %'lli\n", blockNo, cpyAmount);
		/*1/5/11/17 - Copy buf(N * L, L) -> sig[0]. cpyAmount becomes R at the end. N = 0 initially*/
		//fprintf(stderr, "Copy(block, obuf[%'i], %'i)\n", L * blockNo, cpyAmount);
		checkCudaErrors(cudaMemcpy(d_padded_signal, &d_obuf[L * blockNo], cpyAmount * sizeof(float), cudaMemcpyDeviceToDevice));
		if (blockNo != 0) {
			/*6/12/18 - Copy sig(L, M) -> buf[N * L]*/
			//fprintf(stderr, "Copy(obuf[%'i], block[%'i], %'i)\n", L * blockNo, L, M);
			checkCudaErrors(cudaMemcpy(&d_obuf[L * blockNo], &d_padded_signal[L], M * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		
		/*2/7/13/19 - Pad sig(L, M) with 0's, cpyAmount becomes R at the end*/
		
		/*2/7/13/19 - Pad sig(L, M) with 0's, cpyAmount becomes R at the end*/
		//fprintf(stderr, "padZeroes(block, %'i, %'i)\n", cpyAmount, blockSize);
		fillWithZeroes(&d_padded_signal, cpyAmount, blockSize);
		//numBlocks = (blockSize - cpyAmount + numThreads - 1) / numThreads;
		//FillWithZeros<<<numBlocks, numThreads>>>(d_padded_signal, cpyAmount, blockSize);
		
		/*Transform signal*/
		//fprintf(stderr, "FFT block\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_padded_signal, d_sig_complex));
		
		#if defined WIN64 || CALLBACK == 0
			//fprintf(stderr, "NO CALLBACK DOUBLE BLOCK CONVOLUTION\n");
			/*CONVOLUTION*/
			/*3/8/14/20*/
			numBlocks = (blockSize / 2 + numThreads) / numThreads;
			ComplexPointwiseMul << < numBlocks, numThreads >> > (d_sig_complex, d_filter_complex, blockSize / 2 + 1);
			getLastCudaError("Kernel execution failed [ ComplexPointwiseMul]");
		#endif
		/*IFFT*/
		//fprintf(stderr, "Pointwise multiply & IFFT Block\n");
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, d_sig_complex, d_padded_signal));
		if (blockNo != 0) {
			/* 9/15/21 - Point-wise add sig(0,M) + buf[N*L]*/
			//fprintf(stderr, "Add(obuf, block[%'i], %'i)\n", blockNo * L, M);
			PointwiseAdd << <numBlocks, numThreads >> > (d_padded_signal, &d_obuf[blockNo * L], M);
		}
		/*Initial case*/
		if (blockNo == 0) {
			/*4 - Copy sig(0,L) -> buf[0]*/
			//fprintf(stderr, "Copy(obuf, block, %'i)\n", L);
			checkCudaErrors(cudaMemcpy(d_obuf, d_padded_signal, L * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		/*Last case*/
		if (blockNo == blockNum) {
			//fprintf(stderr, "Copy(obuf[%'i], block[%'i], %'i)\n", blockNo * L + M, M, cpyAmount);
			checkCudaErrors(cudaMemcpy(&d_obuf[blockNo * L + M], &d_padded_signal[M], cpyAmount * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		/*Every other case*/
		if(blockNo != 0 && blockNo < blockNum){
			/*10/16 - Copy sig(M, L-M) -> buf[N * L + M]*/
			//fprintf(stderr, "Copy(obuf[%'i], block[%'i], %'i)\n", blockNo * L + M, M, L - M);
			checkCudaErrors(cudaMemcpy(&d_obuf[blockNo * L + M], &d_padded_signal[M], (L - M) * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	//Destroy CUFFT context
	CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(outplan));
	checkCudaErrors(cudaFree(d_padded_signal));
	checkCudaErrors(cudaFree(d_filter_complex));
}

float *blockConvolution(float ** d_ibuf, float ** d_rbuf, long long old_size, long long oFrames, long long audioBlockSize) {
	float *d_obuf = *d_ibuf;
	float *obuf;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * sizeof(float)));
	float minmax = DExtrema(*d_ibuf, old_size);

	blockConvolve(d_ibuf, d_rbuf, old_size, oFrames - old_size + 1);

	float minmax2 = DExtrema(d_obuf, oFrames);

	float scale = minmax/minmax2;

	int blockSize = 128;
	int numBlocks = (oFrames  + blockSize - 1) / blockSize;

    int nStreams = 4;
    int streamSize = (oFrames + nStreams - 1) / nStreams;
    int streamBytes = streamSize * sizeof(float);
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    /*Concurrent copy and pointwise multiply*/
    numBlocks = (streamSize + blockSize - 1) / blockSize;
    for (int i = 0; i < nStreams; ++i) {
    	int offset = i * streamSize;
    	RealFloatScaleConcurrent << < numBlocks, blockSize, 0, stream[i] >> > (d_obuf, oFrames, streamSize, scale, offset);
    	
    	if ( i == nStreams - 1){
    		 streamBytes = (oFrames - offset) * sizeof(float);
    	}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);
	checkCudaErrors(cudaFree(*d_ibuf));

	return obuf;

}
/*Convolution with device memory allocated previously*/
float *convolution(float **d_ibuf, float ** d_rbuf, long long size, long long old_size, long long oFrames) {
	cufftComplex *d_complex;
	float *d_obuf = *d_ibuf;
	float *obuf;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	/*Allocate memory for complex signal & filter*/
	checkCudaErrors(cudaMalloc(&d_complex, (size + 2)* sizeof(cufftComplex)));
	
	checkCudaErrors(cudaMallocHost((void**)&obuf, size * sizeof(float)));
	
	/*Find peak of input signal*/
	float minmax = DExtrema(*d_ibuf, old_size);

	/*Convolving*/
	convolve(d_ibuf, d_rbuf, &d_complex, size);

	/*Find peak of output*/
	float minmax2 = DExtrema(d_obuf, size);
	float scale = minmax/minmax2;
	
	
	/*Block/Thread sizes for kernels*/
	int strides = 1;
	int blockSize = 128;
	int numBlocks = (size + blockSize - 1) / blockSize;
	numBlocks = ( oFrames / strides + blockSize - 1) / blockSize;
	while (numBlocks > (2U << 31 - 1)) {
		numBlocks = ( oFrames  / ++strides + blockSize - 1) / blockSize;
	}
	
	/*Asynchronous copy & scale */
	int nStreams = 4;
	//printf("number of streams: %'i\n", nStreams);
	int streamSize = (oFrames + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);

	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	/*Scale resulting signal according to input signal*/
	numBlocks = (streamSize + blockSize - 1) / blockSize;
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		/*Run scale kernel*/
		RealFloatScaleConcurrent << < numBlocks, blockSize, 0, stream[i] >> > (d_obuf, size, streamSize, scale, offset);
		/*Copy device memory to host asynchronously*/
		if(i == nStreams - 1) streamBytes = sizeof(float) * (oFrames - offset);
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);

	checkCudaErrors(cudaFree(d_obuf));
	return obuf;
}
float *multiGPUFFT(float *ibuf, float *rbuf, long long iFrames, long long rFrames) {
	setlocale(LC_NUMERIC, "");
	long long oFrames = iFrames + rFrames - 1;

	/*get number of devices*/
	int numDevs = 0;
	cudaGetDeviceCount(&numDevs);

	/*Establish all arrays for number of devices*/
	float *d_ibufs[numDevs], *d_rbufs[numDevs];
	cufftComplex *d_Cbufs[numDevs];
	float *d_obuf, *obuf;
	size_t inSizes[numDevs];
	bool doubleBlock = false;
	int amtPerDevice, M = rFrames - 1;
	if( (size_t) oFrames * (size_t)6 > getFreeSize()){
		fprintf(stderr, "ERROR: Device 0 does not have enough memory for thrust operation. Exiting program\n");
		checkCudaErrors(cudaFreeHost(ibuf));
		free(rbuf);
		return NULL;
	}

	/*Find out amount of free memory on each device*/
	long long frames = 0;
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		//most precise is input = freeSize()/16 - 16, but dividing by 32 to conservatively account for cuFFT space
		size_t freeSize = getFreeSize() / 32;
		/*max number of elements that's a power of 2*/
		inSizes[i] = pow(2, floor(log2((double)freeSize)));
		
		
	}

	long long totalAllowedFrames = 0;
	for(int i = 0; i < numDevs; i++){
		totalAllowedFrames += inSizes[i] - M;
	}
	/*Allocating memory for normal case*/
	if (totalAllowedFrames > iFrames){
		for(int i = 0; i < numDevs; i++){
			cudaSetDevice(i);
			if(frames >= iFrames) break;
			frames += inSizes[i] - M;
			checkCudaErrors(cudaMalloc(&d_ibufs[i], inSizes[i] * sizeof(float)));
			checkCudaErrors(cudaMalloc(&d_rbufs[i], inSizes[i] * sizeof(float)));
			checkCudaErrors(cudaMalloc(&d_Cbufs[i], (inSizes[i] + 2) * sizeof(cufftComplex)));
		}
	}
	
	else{

		totalAllowedFrames = 0;
		for(int i = 0; i < numDevs; i++){
			cudaSetDevice(i);
			totalAllowedFrames += getFreeSize() / 4;
			totalAllowedFrames -= rFrames;
		}
		if(totalAllowedFrames < iFrames + M * numDevs){
			fprintf(stderr, "\n\nERROR: NOT ENOUGH COLLETIVE MEMORY ON THE GPUs. EXITING\n\n");
			checkCudaErrors(cudaFreeHost(ibuf));
			free(rbuf);
			return NULL;
		}
		/*Allocating memory for double block case*/
		amtPerDevice = (iFrames + numDevs - 1) / numDevs;
		long long framecount = 0;
		doubleBlock = true;
		for(int i = 0; i < numDevs; i++){
			cudaSetDevice(i);
			//theoretically should be 4. dividing by 8 to be conservative
			size_t freeSize = getFreeSize() / 4;
			freeSize -= rFrames;
			freeSize = pow(2, floor(log2((double)freeSize)));
			int currFrames = amtPerDevice;
			if (currFrames + M > freeSize){
				fprintf(stderr, "WARNING: One GPU has very little memory left. Redistributing memory.\n");
				currFrames = freeSize - M;
				amtPerDevice = iFrames;
			}
			
			if(framecount + currFrames > iFrames){
				currFrames = iFrames - framecount;
			}
			
			if(currFrames == 0){
				inSizes[i] = 0;
				continue;
			}
			inSizes[i] = currFrames + M;
			checkCudaErrors(cudaMalloc(&d_ibufs[i], inSizes[i] * sizeof(float)));
			checkCudaErrors(cudaMalloc(&d_rbufs[i], rFrames * sizeof(float)));
			framecount += currFrames;
			if(framecount >= iFrames) break;
		}
	}
	
	cudaStream_t stream[numDevs];
	/**
	{
	TODO: Peer-to-Peer memcpy of rbuf
	cudaDeviceProp prop;
	for(int i = 0; i )
	checkCudaErrors(cudaGetDeviceProperties(&prop, ))
	int rbufDevNum = 0;
	for(int i = 0; i < numDevs; i++){
		for(int j = 0; j < numDevs; j++){
			if (i == j) continue;
			int num = 0;
			cudaDeviceCanAccessPeer(&num, i, j);
			if(num){
				cudaMemcpyAsync(d_rbufs[i], rbuf, rFrames * sizeof(float));
				rbufDevNum = i;
			}
		}
	}
	**/
	long long blockSize = 512;
	int numBlocks;
	/*Copy each chunk of input into each GPU and pad with 0's*/
	frames = 0;
	//fprintf(stderr, "%s Block\n", doubleBlock ? "Double" : "Single");
	for(int i = 0; i < numDevs; i++){
		cudaSetDevice(i);
		checkCudaErrors(cudaStreamCreate(&stream[i]));
		long long amtRead = inSizes[i] - M;
		if (frames + amtRead > iFrames){
			amtRead = iFrames - frames;
		}
		checkCudaErrors(cudaMemcpyAsync(d_ibufs[i], ibuf + frames, amtRead * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
	
		numBlocks = (inSizes[i] - amtRead - 1 + blockSize) / blockSize;
		FillWithZeros<<<numBlocks, blockSize>>>(d_ibufs[i], amtRead, inSizes[i]);
		//fillWithZeroes(&d_ibufs[i], amtRead, inSizes[i]);
		if(!doubleBlock){	
			//fprintf(stderr, "Filling rbuf with zeroes to pad\n");
			numBlocks = (inSizes[i] - rFrames - 1 + blockSize) / blockSize;
			FillWithZeros<<<numBlocks, blockSize>>>(d_rbufs[i], rFrames, inSizes[i]);
			//fillWithZeroes(&d_rbufs[i], rFrames, inSizes[i]);
		}
		/*WILL BE REPLACED LATER*/
		checkCudaErrors(cudaMemcpyAsync(d_rbufs[i], rbuf, rFrames * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
		//fprintf(stderr, "Copying reverb\n");
		//checkCudaErrors(cudaMemcpy(d_rbufs[i], rbuf, rFrames * sizeof(float), cudaMemcpyHostToDevice));
		///////////////////////////////////////////////////////////
		frames += inSizes[i] - M;
		
		if (frames >= iFrames){
			break;
		}
	}
	checkCudaErrors(cudaFreeHost(ibuf));
	free(rbuf);
	checkCudaErrors(cudaSetDevice(0));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	/*Loop through all input buffers and find the peak*/
	frames = iFrames;
	float minmax1 = 0;
	for(int i = 0 ; i < numDevs; i++){
		cudaSetDevice(i);
		if(frames < 0) break;
		frames -= inSizes[i] - M;
		float minmax = DExtrema(d_ibufs[i], inSizes[i]);
		if(minmax > minmax1)
			minmax1 = minmax;
	}
	/*Convolve all chunks*/
	frames = iFrames;

	if(doubleBlock){
		for(int i = 0; i < numDevs; i++){
			cudaSetDevice(i);
			blockConvolve(&d_ibufs[i], &d_rbufs[i], inSizes[i] - M, rFrames);
		}
	}
	else{
		//fprintf(stderr, "Single Block Convolution\n");
		for(int i = 0; i < numDevs; i++){
			cudaSetDevice(i);
			if(frames < 0) break;
			frames -= inSizes[i] - M;
			convolve(&d_ibufs[i], &d_rbufs[i], &d_Cbufs[i], inSizes[i]);
		}
	}


	/*Overlap-add method to combine the convolved chunks*/
	
	int singleDev = 0;
	// size_t maxFree = 0;
	// for(int i = 0; i < numDevs; i++){
	// 	cudaSetDevice(i);
	// 	if (maxFree < getFreeSize()){
	// 		maxFree = getFreeSize();
	// 		singleDev = i;
	// 	}
	// }
	cudaSetDevice(singleDev);
	float *d_scratchSpace;
	checkCudaErrors(cudaMallocHost(&obuf, oFrames * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_obuf, d_ibufs[0], inSizes[0] * sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMalloc(&d_scratchSpace, M * sizeof(float)));
	cudaSetDevice(0);
	checkCudaErrors(cudaFree(d_ibufs[0]));
	
	long long size = inSizes[0];
	for(int i = 1; i < numDevs; i++){
		long long cpyAmount = inSizes[i] - M;
		if (size + cpyAmount > iFrames) {
			cpyAmount = oFrames - size;
		}
		cudaSetDevice(i);
		checkCudaErrors(cudaMemcpyAsync(d_obuf + size, d_ibufs[i] + M , cpyAmount * sizeof(float), cudaMemcpyDefault, stream[i]));
		checkCudaErrors(cudaMemcpy(d_scratchSpace, d_ibufs[i], M * sizeof(float), cudaMemcpyDefault));
		
		cudaSetDevice(singleDev);
		numBlocks = (M + blockSize - 1) / blockSize;
		PointwiseAdd <<< numBlocks, blockSize, 0, stream[0] >>>(d_scratchSpace, d_obuf + size - M, M);
		
		size += inSizes[i] - M;
		if(size >= oFrames){
			break;
		}	
	}
	frames = iFrames;
	for(int i = 0; i < numDevs; i++){
		if(frames < 0) break;
		frames -= inSizes[i] - M;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaStreamSynchronize(stream[i]));
		checkCudaErrors(cudaStreamDestroy(stream[i]));
		if(i != 0)checkCudaErrors(cudaFree(d_ibufs[i]));
	}
	
	cudaSetDevice(singleDev);
	float minmax2;
	minmax2 = DExtrema(d_obuf, oFrames);
	float scale = minmax1/minmax2;
	
	int strides = 1;
	blockSize = 128;
	numBlocks = (oFrames / strides + blockSize - 1) / blockSize;
	while (numBlocks >(2U << 31 - 1)) {
		numBlocks = (oFrames / ++strides + blockSize - 1) / blockSize;
	}
	
	
	int nStreams = 4;
	int streamSize = (oFrames + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);
	cudaStream_t streams1[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&streams1[i]));
	}

	
	/*Scale + copy 4x*/
	numBlocks = (streamSize + blockSize - 1) / blockSize;
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		RealFloatScaleConcurrent << < numBlocks, blockSize, 0, streams1[i] >> > (d_obuf, oFrames, streamSize, scale, offset);
		if ( i == nStreams - 1){
			streamBytes = (oFrames - offset) * sizeof(float);
		}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, streams1[i]));
	}
	for(int i = 0; i < 4; i++){
		checkCudaErrors(cudaStreamDestroy(streams1[i]));
	}
	checkCudaErrors(cudaSetDevice(0));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr,"Time for GPU convolution: %f ms\n", milliseconds);
	checkCudaErrors(cudaFree(d_obuf));
	return obuf;

}
