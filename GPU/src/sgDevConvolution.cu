#include "Convolution.cuh"
// Define the device pointer to the callback routine. The host code will fetch this and pass it to CUFFT
#ifndef WIN64
__device__ cufftCallbackLoadC myOwnCallbackPtr = cbComplexPointwiseMul;
#endif
void findBlockSize(long long iFrames, int M, size_t *blockSize, int *blockNum) {
	/*Finding block size/number*/

	int myExp = ceil(log2((float)(iFrames + M)));
	while (pow(2, myExp) > INT_MAX) {
		myExp--;
	}
	size_t smallerBlockSize = pow(2, myExp);
	*blockNum = 1;
	size_t workspace;
	CHECK_CUFFT_ERRORS(cufftEstimate1d(smallerBlockSize, CUFFT_R2C, 2, &workspace));

	/*Look for block size worth with 2 complex arrays
	Multiply by 4 to leave some room*/
	while (getFreeSize() < workspace + (smallerBlockSize / 2 + 1) * 8L * 4L) {
		myExp--;
		smallerBlockSize = pow(2, myExp);
		(*blockNum)++;
		CHECK_CUFFT_ERRORS(cufftEstimate1d(smallerBlockSize, CUFFT_R2C, 2, &workspace));
	}

	fprintf(stderr, "blockSize: %i\t numBlocks: %i\n", smallerBlockSize, *blockNum);
	*blockSize = smallerBlockSize;
}

void mismatchedConvolve(passable *p) {
	flags flag = p->type;
	long long paddedSize = p->paddedSize;
	float *d_ibuf = p->input->d_buf;
	float *d_rbuf = p->reverb->d_buf;

	/*Create forward FFT plan*/
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, paddedSize, CUFFT_R2C, 1));

	/*Create inverse FFT plan*/
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, paddedSize, CUFFT_C2R, 1));

	/*Transform Input Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_ibuf, (cufftComplex*)d_ibuf));
	if (flag == stereo_mono) {
		Print("Transforming Ch 2 of input\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_ibuf + paddedSize, (cufftComplex*)d_ibuf + paddedSize / 2 + 1));
	}
	/*Transform Filter Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_rbuf, (cufftComplex*)d_rbuf));
	if (flag == mono_stereo) {
		Print("Transforming Ch 2 of reverb\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_rbuf + paddedSize, (cufftComplex*)d_rbuf + paddedSize / 2 + 1));
	}
#if defined WIN64 || CB == 0
	/*NO CB VERSION*/

	/*CONVOLUTION*/
	int blockSize = 256;
	int numBlocks = (paddedSize / 2 + 1 + blockSize - 1) / blockSize;
	if (flag == mono_stereo) {
		Print("Convolving & Inverse Transforming for stereo reverb\n");
		ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_rbuf, (cufftComplex*)d_ibuf, paddedSize / 2 + 1);
		ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_rbuf + paddedSize / 2 + 1, (cufftComplex*)d_ibuf, paddedSize / 2 + 1);
	}
	else {
		ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_ibuf, (cufftComplex*)d_rbuf, paddedSize / 2 + 1);
		ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_ibuf + paddedSize / 2 + 1, (cufftComplex*)d_rbuf, paddedSize / 2 + 1);
	}
#else
	/*Copy over the host copy of callback function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));

	/*Associate the load callback with the plan*/
	if (flag == stereo_mono) {
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
			(void **)&d_rbuf));
	}
	else {
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
			(void **)&d_ibuf));
	}
#endif
	if (flag == stereo_mono) {
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_ibuf, d_ibuf));
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_ibuf + paddedSize / 2 + 1, d_ibuf + paddedSize));
	}
	else {
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_rbuf, d_rbuf));
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_rbuf + paddedSize / 2 + 1, d_rbuf + paddedSize));
	}
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(outplan));
}

void convolve(float *d_ibuf, float *d_rbuf, long long paddedSize) {
	/*Create forward FFT plan*/
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, paddedSize, CUFFT_R2C, 1));

	/*Create inverse FFT plan*/
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, paddedSize, CUFFT_C2R, 1));

	/*Transform Complex Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_ibuf, (cufftComplex*)d_ibuf));

	/*Transform Filter Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_rbuf, (cufftComplex*)d_rbuf));

#if defined WIN64 || CB == 0
	/*NO CB VERSION*/
	/*CONVOLUTION*/
	int blockSize = 256;
	int numBlocks = (paddedSize + blockSize - 1) / blockSize;

	ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_ibuf, (cufftComplex*)d_rbuf, paddedSize / 2 + 1);
	getLastCudaError("Kernel execution failed [ ComplexPointwiseMul]");
#else
	/*Copy over the host copy of callback function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr,
		sizeof(hostCopyOfCallbackPtr)));

	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
		(void **)&d_rbuf));

#endif
	CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_ibuf, (cufftReal*)d_ibuf));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(outplan));
}
/*Assumes that d_buf contains paddedSize * 2 elements.
Input is in first half, filter is in second half, and both are padded*/
void convolveBatched(float *d_buf, long long paddedSize) {
	float *d_rbuf = d_buf + paddedSize + 2;
	/*Create forward FFT plan*/
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	/*cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
		int *inembed, int istride, int idist,
		int *onembed, int ostride, int odist,
		cufftType type, int batch);*/
		/*stride = skip length. Ex 1 = every element, 2 = every other element*/
			/*use for interleaving???*/
		/*idist/odist is space between batches of transforms*/
			/*need to check if odist is in terms of complex numbers or floats*/
		/*inembed/onembed are for 2D/3D, num elements per dimension*/
	int n = paddedSize;
	CHECK_CUFFT_ERRORS(
		cufftPlanMany(&plan, 1, &n,
			&n, 1, n + 2,
			&n, 1, n / 2 + 1,
			CUFFT_R2C, 2)
	)

		/*Create inverse FFT plan*/
		cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, paddedSize, CUFFT_C2R, 1));

	/*Transform Complex Signal*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_buf, (cufftComplex*)d_buf));

#if defined WIN64 || CB == 0
	/*NO CB VERSION*/
	/*CONVOLUTION*/
	int blockSize = 256;
	int numBlocks = (paddedSize + blockSize - 1) / blockSize;

	ComplexPointwiseMul << < numBlocks, blockSize >> > ((cufftComplex*)d_buf, (cufftComplex*)d_rbuf, paddedSize / 2 + 1);
	getLastCudaError("Kernel execution failed [ ComplexPointwiseMul]");
#else
	/*Copy over the host copy of callback function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr,
		sizeof(hostCopyOfCallbackPtr)));

	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
		(void **)&d_rbuf));

#endif
	CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_buf, (cufftReal*)d_buf));
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(outplan));
}

void overlapAdd(float *d_ibuf, cufftComplex *d_rbuf, long long iFrames, long long M,
	long long blockSize, int blockNum, cufftHandle plan, cufftHandle outplan) {
	float *d_block;
	long long L = blockSize - M;

	int numThreads = 256;
	int numBlocks = (M + numThreads - 1) / numThreads;

	checkCudaErrors(cudaMalloc(&d_block, (blockSize / 2 + 1) * sizeof(cufftComplex)));

	for (int blockNo = 0; blockNo < blockNum; blockNo++) {
		long long cpyAmount = L;
		if (blockNo == blockNum && iFrames != cpyAmount) {
			cpyAmount = iFrames % L;
		}
		/*1/5/11/17 - Copy buf(N * L, L) -> sig[0]. cpyAmount becomes R at the end. N = 0 initially*/
		//fprintf(stderr, "Copy(block, obuf[%'i], %'i)\n", L * blockNo, cpyAmount);
		checkCudaErrors(cudaMemcpy(d_block, &d_ibuf[L * blockNo], cpyAmount * sizeof(float), cudaMemcpyDeviceToDevice));
		if (blockNo != 0) {
			/*6/12/18 - Copy sig(L, M) -> buf[N * L]*/
			//fprintf(stderr, "Copy(obuf[%'i], block[%'i], %'i)\n", L * blockNo, L, M);
			checkCudaErrors(cudaMemcpy(&d_ibuf[L * blockNo], &d_block[L], M * sizeof(float), cudaMemcpyDeviceToDevice));
		}

		/*2/7/13/19 - Pad sig(L, M) with 0's, cpyAmount becomes R at the end*/
		fillWithZeroes(&d_block, cpyAmount, blockSize);

		/*Transform signal*/
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_block, (cufftComplex*)d_block));

#if defined WIN64 || CB == 0
		/*CONVOLUTION*/
		/*3/8/14/20*/
		numBlocks = (blockSize / 2 + numThreads) / numThreads;
		ComplexPointwiseMul << < numBlocks, numThreads >> > ((cufftComplex*)d_block,
			(cufftComplex*)d_rbuf, blockSize / 2 + 1);
		getLastCudaError("Kernel execution failed [ ComplexPointwiseMul]");
#endif
		/*IFFT*/
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_block, (cufftReal*)d_block));
		if (blockNo != 0) {
			/* 9/15/21 - Point-wise add sig(0,M) + buf[N*L]*/
			PointwiseAdd << <numBlocks, numThreads >> > ((float*)d_block, &d_ibuf[blockNo * L], M);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		/*Corner case where only one block*/
		if (blockNo == 0 && blockNo == blockNum - 1) {
			checkCudaErrors(cudaMemcpy(d_ibuf, d_block, (cpyAmount + M) * sizeof(float), cudaMemcpyDeviceToDevice));
			break;
		}
		/*Initial case*/
		if (blockNo == 0) {
			/*4 - Copy sig(0,L) -> buf[0]*/
			checkCudaErrors(cudaMemcpy(d_ibuf, d_block, L * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		/*Last case*/
		if (blockNo == blockNum - 1) {
			//fprintf(stderr, "Copy(obuf[%'i], block[%'i], %'i)\n", blockNo * L + M, M, cpyAmount);
			checkCudaErrors(cudaMemcpy(&d_ibuf[blockNo * L + M], &d_block[M], cpyAmount * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		/*Every other case*/
		if (blockNo != 0 && blockNo < blockNum) {
			/*10/16 - Copy sig(M, L-M) -> buf[N * L + M]*/
			checkCudaErrors(cudaMemcpy(&d_ibuf[blockNo * L + M], &d_block[M], (L - M) * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	checkCudaErrors(cudaFree(d_block));
}

float *blockConvolution(passable *p) {
	float *d_ibuf = p->input->d_buf;
	float *rbuf = p->reverb->buf;
	cufftComplex *d_filter_complex;
	float *d_obuf = d_ibuf, *obuf;
	long long rFrames = p->reverb->frames;
	long long iFrames = p->input->frames;
	long long oFrames = rFrames + iFrames - 1;
	flags flag = p->type;
	int oCh = flag == mono_mono ? 1 : 2;
	float minmax, minmax2;
	cudaEvent_t start, stop;


	int M = rFrames - 1;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	size_t blockSize = iFrames;
	int blockNum = 0;

	/*Find block size and store in blockSize and blockNum*/
	findBlockSize(iFrames, M, &blockSize, &blockNum);

	/*Allocating memory for output*/
	Print("Allocating memory for output\n");
	checkCudaErrors(cudaMallocHost((void**)&obuf, oFrames * oCh * sizeof(float)));


	/*Find peak of input signal*/
	Print("Finding peak of input signal\n");
	minmax = DExtrema(d_ibuf, oFrames * p->input->channels);

	/*TRANSFORMING FILTER*/
	/*Allocating Memory*/
	Print("Allocating memory\n");
	int ch = p->reverb->channels;
	checkCudaErrors(cudaMalloc(&d_filter_complex, (blockSize / 2 + 1) * ch * sizeof(cufftComplex)));

	/*Block/Thread sizes for kernels*/
	int numThreads = 256;
	int numBlocks = (blockSize + 2 - rFrames + numThreads - 1) / numThreads;
	cudaStream_t stream[4];
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	/* Copy over filter */
	Print("Copying over filter\n");
	FillWithZeros << <numBlocks, numThreads, 0, stream[0] >> > ((float*)d_filter_complex, rFrames, blockSize + 2);
	if (ch == 2) {
		FillWithZeros << <numBlocks, numThreads, 0, stream[1] >> > ((float*)d_filter_complex + blockSize + 2,
			rFrames, blockSize * 2 + 4);
		checkCudaErrors(cudaMemcpyAsync((float*)d_filter_complex + blockSize + 2,
			rbuf + rFrames, rFrames * sizeof(float), cudaMemcpyHostToDevice, stream[2]));
	}
	checkCudaErrors(cudaMemcpyAsync((float*)d_filter_complex, rbuf,
		rFrames * sizeof(float), cudaMemcpyHostToDevice, stream[3]));


	/*Create cuFFT plan*/
	Print("Creating FFT plans\n");
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, blockSize, CUFFT_R2C, 1));

	/*Plans*/
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, blockSize, CUFFT_C2R, 1));


#if defined WIN64 || CB == 0
#else
	/*Create host pointer to CB Function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));

	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr,
		CUFFT_CB_LD_COMPLEX, (void **)&d_filter_complex));
#endif	

	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamSynchronize(stream[i]));
	}

	checkCudaErrors(cudaFreeHost(rbuf));

	/*Transform Filter*/
	Print("Transforming filter\n");
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_filter_complex, (cufftComplex*)d_filter_complex));
	if (ch == 2) {
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_filter_complex + blockSize + 2, (cufftComplex*)d_filter_complex + blockSize / 2 + 1));
	}

	/*Convolving*/
	if (flag == mono_mono) {
		Print("mono_mono Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else if (flag == stereo_stereo) {
		Print("stereo_stereo Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex + blockSize / 2 + 1,
			iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else if (flag == stereo_mono) {
		Print("stereo_mono Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else {
		Print("mono_stereo Convolving\n");
		checkCudaErrors(cudaMemcpy(d_obuf + oFrames, d_obuf, oFrames * sizeof(float), cudaMemcpyDeviceToDevice));
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex + blockSize / 2 + 1,
			iFrames, M, blockSize, blockNum, plan, outplan);
	}
	checkCudaErrors(cudaFree(d_filter_complex));
	CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(outplan));

	/*Find peak of output*/
	Print("Find peak of output\n");
	minmax2 = DExtrema(d_obuf, oFrames * oCh);

	float scale = minmax / minmax2;
	long long end = oFrames * oCh;
	fprintf(stderr, "end: %lli\n", end);
	/*Block/Thread sizes for kernels*/
	blockSize = 512;
	numBlocks = (end + blockSize - 1) / blockSize;
	// RealFloatScale << < numBlocks, blockSize>> > (d_obuf, end, scale);
	// checkCudaErrors(cudaMemcpy(obuf, d_obuf, end * sizeof(float), cudaMemcpyDeviceToHost));

	/*Asynchronous copy & scale */
	const int nStreams = 4;
	int streamSize = (end + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);
	numBlocks = (streamSize + blockSize - 1) / blockSize;

	Print("Scaling and copying\n");
	for (int i = 0; i < nStreams; ++i) {
		long long offset = i * streamSize;
		/*Run scale kernel*/
		RealFloatScaleConcurrent << < numBlocks, blockSize, 0, stream[i] >> > (d_obuf, end, streamSize, scale, offset);
		/*Copy device memory to host asynchronously*/
		if (i == nStreams - 1) {
			streamBytes = sizeof(float) * (end - offset);
		}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for GPU convolution: %f ms\n", milliseconds);

	checkCudaErrors(cudaFree(d_obuf));
	return obuf;
}
float *convolution(passable *p) {
	float *d_ibuf = p->input->d_buf;
	float *d_rbuf = p->reverb->d_buf;
	float *d_obuf = d_ibuf;
	float *obuf;
	flags flag = p->type;
	int oCh = flag == mono_mono ? 1 : 2;
	long long paddedSize = p->paddedSize;
	float minmax, minmax2;
	cudaEvent_t start, stop;
	//printMe(p);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	/*Allocating host memory for output*/
	Print("Allocating host memory for output\n");
	checkCudaErrors(cudaMallocHost((void**)&obuf, paddedSize * oCh * sizeof(float)));

	/*Find peak of input signal*/
	Print("Finding peak of input signal\n");
	minmax = DExtrema(d_ibuf, paddedSize * oCh);

	/*Convolving*/
	if (flag == mono_mono) {
		Print("mono_mono Convolving\n");
		convolve(d_ibuf, d_rbuf, paddedSize);
		//convolveBatched(d_ibuf, paddedSize); // not doing batched because it is very slightly slower (~20 ms)
	}
	else if (flag == stereo_stereo) {
		Print("stereo_stereo Convolving\n");
		convolve(d_ibuf, d_rbuf, paddedSize);
		convolve(d_ibuf + paddedSize, d_rbuf + paddedSize, paddedSize);
	}
	else {
		mismatchedConvolve(p);
		if (flag == mono_stereo) {
			d_obuf = d_rbuf;
		}
	}

	/*Find peak of output*/
	Print("Find peak of output\n");
	minmax2 = DExtrema(d_obuf, paddedSize * oCh);

	float scale = minmax / minmax2;
	long long end = paddedSize * oCh;

	/*Block/Thread sizes for kernels*/
	int blockSize = 512;
	int numBlocks = (end + blockSize - 1) / blockSize;

	/*Asynchronous copy & scale */
	const int nStreams = 4;
	int streamSize = (end + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);
	numBlocks = (streamSize + blockSize - 1) / blockSize;

	/*Create streams*/
	Print("Creating streams\n");
	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	Print("Scaling and copying\n");
	for (int i = 0; i < nStreams; ++i) {
		long long offset = i * streamSize;
		/*Run scale kernel*/
		RealFloatScaleConcurrent << < numBlocks, blockSize, 0, stream[i] >> > (d_obuf, end, streamSize, scale, offset);
		/*Copy device memory to host asynchronously*/
		if (i == nStreams - 1) {
			streamBytes = sizeof(float) * (end - offset);
		}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for GPU convolution: %f ms\n", milliseconds);

	checkCudaErrors(cudaFree(d_ibuf));
	checkCudaErrors(cudaFree(d_rbuf));
	return obuf;
}
void blockProcess(passable* p) {
	float* d_ibuf = p->input->d_buf;
	float* rbuf = p->reverb->buf;
	cufftComplex* d_filter_complex;
	float* d_obuf = d_ibuf;
	long long rFrames = p->reverb->frames;
	long long iFrames = p->input->frames;
	long long oFrames = rFrames + iFrames - 1;
	flags flag = p->type;
	int M = rFrames - 1;
	size_t blockSize = iFrames;
	int blockNum = 0;

	/*Find block size and store in blockSize and blockNum*/
	findBlockSize(iFrames, M, &blockSize, &blockNum);

	/*TRANSFORMING FILTER*/
	/*Allocating Memory*/
	Print("Allocating memory\n");
	int ch = p->reverb->channels;
	checkCudaErrors(cudaMalloc(&d_filter_complex, (blockSize / 2 + 1) * ch * sizeof(cufftComplex)));

	/*Block/Thread sizes for kernels*/
	int numThreads = 256;
	int numBlocks = (blockSize + 2 - rFrames + numThreads - 1) / numThreads;
	cudaStream_t stream[4];
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	/* Copy over filter */
	Print("Copying over filter\n");
	FillWithZeros << <numBlocks, numThreads, 0, stream[0] >> > ((float*)d_filter_complex, rFrames, blockSize + 2);
	if (ch == 2) {
		FillWithZeros << <numBlocks, numThreads, 0, stream[1] >> > ((float*)d_filter_complex + blockSize + 2,
			rFrames, blockSize * 2 + 4);
		checkCudaErrors(cudaMemcpyAsync((float*)d_filter_complex + blockSize + 2,
			rbuf + rFrames, rFrames * sizeof(float), cudaMemcpyHostToDevice, stream[2]));
	}
	checkCudaErrors(cudaMemcpyAsync((float*)d_filter_complex, rbuf,
		rFrames * sizeof(float), cudaMemcpyHostToDevice, stream[3]));


	/*Create cuFFT plan*/
	Print("Creating FFT plans\n");
	cufftHandle plan;
	CHECK_CUFFT_ERRORS(cufftCreate(&plan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, blockSize, CUFFT_R2C, 1));

	/*Plans*/
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, blockSize, CUFFT_C2R, 1));


#if defined WIN64 || CB == 0
#else
	/*Create host pointer to CB Function*/
	cufftCallbackLoadC hostCopyOfCallbackPtr;
	checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));

	/*Associate the load callback with the plan*/
	CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void**)&hostCopyOfCallbackPtr,
		CUFFT_CB_LD_COMPLEX, (void**)&d_filter_complex));
#endif	

	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamSynchronize(stream[i]));
	}

	checkCudaErrors(cudaFreeHost(rbuf));

	/*Transform Filter*/
	Print("Transforming filter\n");
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_filter_complex, (cufftComplex*)d_filter_complex));
	if (ch == 2) {
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_filter_complex + blockSize + 2, (cufftComplex*)d_filter_complex + blockSize / 2 + 1));
	}
	/*Convolving*/
	if (flag == mono_mono) {
		Print("mono_mono Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else if (flag == stereo_stereo) {
		Print("stereo_stereo Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex + blockSize / 2 + 1,
			iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else if (flag == stereo_mono) {
		Print("stereo_mono Convolving\n");
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
	}
	else {
		Print("mono_stereo Convolving\n");
		checkCudaErrors(cudaMemcpy(d_obuf + oFrames, d_obuf, oFrames * sizeof(float), cudaMemcpyDeviceToDevice));
		overlapAdd(d_obuf, d_filter_complex, iFrames, M, blockSize, blockNum, plan, outplan);
		overlapAdd(d_obuf + oFrames, d_filter_complex + blockSize / 2 + 1,
			iFrames, M, blockSize, blockNum, plan, outplan);
	}
	checkCudaErrors(cudaFree(d_filter_complex));
	CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(outplan));
}

void convolutionPicker(passable* p) {
	
}
void process(passable* p) {
	float* d_ibuf = p->input->d_buf;
	float* d_rbuf = p->reverb->d_buf;
	float* d_obuf = d_ibuf;
	long long paddedSize = p->paddedSize;
	flags flag = p->type;

	/*Convolving*/
	if (flag == mono_mono) {
		Print("mono_mono Convolving\n");
		//convolve(d_ibuf, d_rbuf, paddedSize);
		convolveBatched(d_ibuf, paddedSize);
	}
	else if (flag == stereo_stereo) {
		Print("stereo_stereo Convolving\n");
		convolve(d_ibuf, d_rbuf, paddedSize);
		convolve(d_ibuf + paddedSize, d_rbuf + paddedSize, paddedSize);
	}
	else {
		mismatchedConvolve(p);
		if (flag == mono_stereo) {
			d_obuf = d_rbuf;
		}
	}
}
void asyncCopyScale(passable* p, float *obuf, long long end, float scale) {
	float* d_obuf = p->input->d_buf;
	/*Block/Thread sizes for kernels*/
	int blockSize = 512;
	int numBlocks = (end + blockSize - 1) / blockSize;

	/*Asynchronous copy & scale */
	const int nStreams = 4;
	int streamSize = (end + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);
	numBlocks = (streamSize + blockSize - 1) / blockSize;

	/*Create streams*/
	Print("Creating streams\n");
	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	Print("Scaling and copying\n");
	for (int i = 0; i < nStreams; ++i) {
		long long offset = i * streamSize;
		/*Run scale kernel*/
		RealFloatScaleConcurrent << < numBlocks, blockSize, 0, stream[i] >> > (d_obuf, end, streamSize, scale, offset);
		/*Copy device memory to host asynchronously*/
		if (i == nStreams - 1) {
			streamBytes = sizeof(float) * (end - offset);
		}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}
}
float* convolutionWrapper(passable* p, bool blockProcessingOn) {
	float* d_ibuf = p->input->d_buf;
	float* d_rbuf = p->reverb->d_buf;
	float* d_obuf = d_ibuf;
	float* obuf;
	flags flag = p->type;
	int oCh = flag == mono_mono ? 1 : 2;
	long long paddedSize = p->paddedSize;
	float minmax, minmax2;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	/*Allocating host memory for output*/
	Print("Allocating host memory for output\n");
	checkCudaErrors(cudaMallocHost((void**)&obuf, paddedSize * oCh * sizeof(float)));

	/*Find peak of input signal*/
	Print("Finding peak of input signal\n");
	minmax = DExtrema(d_ibuf, paddedSize * oCh);

	/*Performing Convolution*/
	if (blockProcessingOn) {
		blockProcess(p);
	}
	else {
		process(p);
	}

	/*Find peak of output*/
	Print("Find peak of output\n");
	minmax2 = DExtrema(d_obuf, paddedSize * oCh);

	float scale = minmax / minmax2;
	long long end = paddedSize * oCh;

	asyncCopyScale(p, obuf, end, scale);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for GPU convolution: %f ms\n", milliseconds);

	checkCudaErrors(cudaFree(d_ibuf));
	checkCudaErrors(cudaFree(d_rbuf));
	return obuf;
}