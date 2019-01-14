#include "Convolution.cuh"

extern __device__ cufftCallbackLoadC myOwnCallbackPtr;
void printMGInfo(multi_gpu_struct m){
	fprintf(stderr, "d_ibuf    : %p\n", m.d_ibuf);
	fprintf(stderr, "d_rbuf    : %p\n", m.d_rbuf);
	fprintf(stderr, "size      : %'lli\n", m.size);
	fprintf(stderr, "L         : %'lli\n", m.L);
	fprintf(stderr, "offset    : %'lli\n", m.offset);
	fprintf(stderr, "blockSize : %'lu\n", m.blockSize);
	fprintf(stderr, "numBlocks : %i\n", m.numBlocks);
	fprintf(stderr, "M         : %i\n\n", m.M);
}
int findLargestGPU(int numDevs) {
	/*Find device with largest free space*/
	size_t maxFree = 0;
	int singleDev = 0;
	for (int i = 0; i < numDevs; i++) {
		checkCudaErrors(cudaSetDevice(i));
		size_t size = getFreeSize();
		if (maxFree > size) {
			maxFree = size;
			singleDev = i;
		}
	}
	return singleDev;
}
/*Assumes device is set by caller*/
void scaling(float *obuf, float *d_obuf, float scale, long long oFrames) {
	Print("Scaling and copying out\n");
	int T = 128;
	int B = (oFrames + T - 1) / T;
	const int nStreams = 4;
	cudaStream_t stream[nStreams];
	int streamSize = (oFrames + nStreams - 1) / nStreams;
	int streamBytes = streamSize * sizeof(float);


	/*Scale + copy 4x*/
	B = (streamSize + T - 1) / T;
	for (int i = 0; i < nStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
		int offset = i * streamSize;
		RealFloatScaleConcurrent << < B, T, 0, stream[i] >> > (d_obuf, oFrames, streamSize, scale, offset);
		if (i == nStreams - 1) {
			streamBytes = (oFrames - offset) * sizeof(float);
		}
		checkCudaErrors(cudaMemcpyAsync(&obuf[offset], &d_obuf[offset], streamBytes,
			cudaMemcpyDeviceToHost, stream[i]));
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamSynchronize(stream[i]));
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
	checkCudaErrors(cudaFree(d_obuf));
}
void fillBuffers(int numDevs, cudaStream_t *stream, bool sg, multi_gpu_struct *mg, passable *p){
	float *ibuf = p->input->buf;
	float *rbuf = p->reverb->buf;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	/*Fill and pad buffers*/
	Print("Filling and padding buffers\n");
	int T = 512;
	int B;
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		checkCudaErrors(cudaSetDevice(i));
		long long inputEnd;
		if(sg){
			 inputEnd = mg[i].blockSize;
		}
		else {
			inputEnd = mg[i].size + M;
		}
		long long reverbEnd = mg[i].blockSize;

		/*Copy input*/
		checkCudaErrors(cudaMemcpyAsync(mg[i].d_ibuf, ibuf + mg[i].offset, mg[i].size * sizeof(float),
			cudaMemcpyHostToDevice, stream[i * 4]));

		/*Pad input*/
		B = (inputEnd + 2 - mg[i].size + T - 1) / T;
		FillWithZeros << <B, T, 0, stream[i * 4 + 1] >> > (mg[i].d_ibuf, mg[i].size, inputEnd + 2);

		/*Pad reverb*/
		B = (reverbEnd + 2 - rFrames + T) / T;
		FillWithZeros << <B, T, 0, stream[i * 4 + 2] >> > (mg[i].d_rbuf, rFrames, reverbEnd + 2);

		/*Copy reverb*/
		checkCudaErrors(cudaMemcpyAsync(mg[i].d_rbuf, rbuf, rFrames * sizeof(float),
			cudaMemcpyHostToDevice, stream[i * 4 + 3]));
	}
	checkCudaErrors(cudaFreeHost(ibuf));
	checkCudaErrors(cudaFreeHost(rbuf));
}
float * overlapAddMultiGPU(multi_gpu_struct *mg, passable *p, int numDevs, cudaStream_t *stream, int singleDev) {
	int M = p->reverb->frames - 1;
	long long oFrames = p->input->frames + M;
	int streamsPerDev = 4;
	/*Overlap-add method to combine the convolved chunks*/
	Print("Overlap-add Reconstruction\n");

	checkCudaErrors(cudaSetDevice(singleDev));
	float *d_obuf, *d_scratchSpace;

	Print("Allocating output buffer\n");
	checkCudaErrors(cudaMalloc(&d_obuf, oFrames * sizeof(float)));
	Print("Allocating scratch space\n");
	checkCudaErrors(cudaMalloc(&d_scratchSpace, M * (numDevs - 1) * sizeof(float)));
	checkCudaErrors(cudaMemcpyAsync(d_obuf, mg[0].d_ibuf, (mg[0].size + M) * sizeof(float),
		cudaMemcpyDefault, stream[singleDev * streamsPerDev]));
	int T = 512;
	int B = (M + T - 1) / T;
	for (int i = 1; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		cudaSetDevice(i);
		checkCudaErrors(cudaMemcpyAsync(d_scratchSpace + (M * (i - 1)), mg[i].d_ibuf, M * sizeof(float),
			cudaMemcpyDefault, stream[i * streamsPerDev]));
		checkCudaErrors(cudaMemcpyAsync(d_obuf + mg[i].offset + M, mg[i].d_ibuf + M,
			mg[i].size * sizeof(float), cudaMemcpyDefault, stream[i * streamsPerDev + 1]));
	}
	int count = 0;
	for (int i = 1; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		cudaSetDevice(i);
		checkCudaErrors(cudaStreamSynchronize(stream[i * streamsPerDev]));
		checkCudaErrors(cudaStreamSynchronize(stream[i * streamsPerDev + 1]));
		cudaSetDevice(singleDev);
		PointwiseAdd << < B, T, 0, stream[singleDev * streamsPerDev + count] >> > (d_scratchSpace + (M * (i - 1)), d_obuf + mg[i].offset, M);
		if (count < 4) count++;
	}

	checkCudaErrors(cudaFree(d_scratchSpace));
	return d_obuf;
}
/*
typedef struct multi_gpu_struct{
	float *d_ibuf;
	float *d_rbuf;
	long long size;
	size_t blockSize;
	long long L;
	long long offset;
	int numBlocks;
} multi_gpu_struct;
*/
float singleBlockMethodOne(passable *p, int numDevs, cudaStream_t *stream, multi_gpu_struct *mg) {
	long long iFrames = p->input->frames;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	/*Filling in data*/
	Print("Filling in data\n");
	size_t inChunk = (iFrames + numDevs - 1) / numDevs;
	size_t blockSize = pow(2, ceil(log2((double)(inChunk + M))));
	for (int i = 0; i < numDevs; i++) {
		mg[i].offset = inChunk * i;
		mg[i].blockSize = blockSize;
		mg[i].numBlocks = 0;
		if (i == numDevs - 1) {
			mg[i].L = iFrames - mg[i].offset;
			mg[i].size = mg[i].L;
		}
		else {
			mg[i].L = inChunk;
			mg[i].size = mg[i].L;
		}
	}

	/*Allocate memory*/
	Print("Allocating memory\n");
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		checkCudaErrors(cudaMalloc(&(mg[i].d_ibuf), (blockSize + 2 ) * sizeof(cufftComplex)));
		mg[i].d_rbuf = mg[i].d_ibuf + blockSize + 2;
	}
	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
	}
	
	fillBuffers(numDevs, stream, true, mg, p);

	/*Finding peak*/
	float minmax1 = 0;
	Print("Finding input peak\n");
	for (int i = 0; i < numDevs; i++) {
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaStreamSynchronize(stream[i * 4]));
		float minmax = DExtrema(mg[i].d_ibuf, mg[i].size);
		if (minmax > minmax1)
			minmax1 = minmax;
	}

	/*Convolving*/
	Print("Convolving\n");
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		convolveBatched(mg[i].d_ibuf, mg[i].blockSize);
	}
	return minmax1;

}
float singleBlockMethodTwo(passable *p, int numDevs, size_t *freeSizes, cudaStream_t *stream, multi_gpu_struct *mg) {
	long long iFrames = p->input->frames;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	/*Filling in data*/
	Print("Filling in data\n");
	long long framecount = 0;
	for (int i = 0; i < numDevs; i++) {

		mg[i].M = (int) M;
		mg[i].numBlocks = 0;
		mg[i].L = 0;
		if (framecount == iFrames) {
			mg[i].size = 0;
			mg[i].offset = -1;
			mg[i].blockSize = 0;
			continue;
		}
		/*Estimated amount of allocatable floats*/
		size_t actualFreeSize = freeSizes[i] / 4 / 2;

		/*Estimated amount of allocatable floats to the power of 2*/
		size_t blockSize = pow(2, floor(log2((double)actualFreeSize)));

		/*Number of elements in input/reverb chunk*/
		size_t chunk = blockSize / 2;
		
		/*Number of elements in input - M*/
		chunk -= M;

		mg[i].offset = framecount;
		mg[i].size = chunk < (iFrames - framecount) ? (long long) chunk : (long long) (iFrames - framecount);
		mg[i].blockSize = pow(2, ceil(log2((double)mg[i].size)));
		framecount += mg[i].size;
	}

	/*Allocate memory*/
	Print("Allocating memory\n");
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		cudaSetDevice(i);
		checkCudaErrors(cudaMalloc(&(mg[i].d_ibuf), (mg[i].blockSize / 2 + 1) * 2 * sizeof(cufftComplex)));
		mg[i].d_rbuf = mg[i].d_ibuf + mg[i].blockSize + 2;
	}

	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
	}
	
	fillBuffers(numDevs, stream, true, mg, p);

	/*Finding peak*/
	float minmax1 = 0;
	Print("Finding input peak\n");
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaStreamSynchronize(stream[i * 4]));
		float minmax = DExtrema(mg[i].d_ibuf, mg[i].size);
		if (minmax > minmax1)
			minmax1 = minmax;
	}

	/*Convolving*/
	Print("Convolving\n");
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		cudaSetDevice(i);
		convolveBatched(mg[i].d_ibuf, mg[i].blockSize);
	}
	return minmax1;
}
float doubleBlockMethodOne(passable *p, int numDevs, cudaStream_t *stream, multi_gpu_struct *mg) {
	long long iFrames = p->input->frames;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	/*Filling in data*/
	Print("Filling in data\n");
	size_t outChunk = (iFrames + numDevs - 1) / numDevs;
	for (int i = 0; i < numDevs; i++) {
		mg[i].offset = outChunk * i;
		mg[i].size = i == numDevs - 1 ? iFrames - mg[i].offset : outChunk;
	}

	/*Allocate memory*/
	Print("Allocating Memory & Finding block sizes\n");
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		checkCudaErrors(cudaMalloc(&(mg[i].d_ibuf), (mg[i].size + M) * sizeof(float)));
		findBlockSize(mg[i].size, M, &(mg[i].blockSize), &(mg[i].numBlocks));
		checkCudaErrors(cudaMalloc(&(mg[i].d_rbuf), (mg[i].blockSize / 2 + 1) * sizeof(cufftComplex)));
		mg[i].L = mg[i].blockSize - M;
	}

	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
	}
	
	fillBuffers(numDevs, stream, false, mg, p);


	/*Finding peak*/
	float minmax1 = 0;
	Print("Finding input peak\n");
	for (int i = 0; i < numDevs; i++) {
		checkCudaErrors(cudaSetDevice(i));
		float minmax = DExtrema(mg[i].d_ibuf, mg[i].size);
		if (minmax > minmax1)
			minmax1 = minmax;
	}

	/*Convolving*/
	Print("Convolving\n");
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		/*Create cuFFT plan*/
		Print("Creating FFT plans\n");
		cufftHandle plan;
		CHECK_CUFFT_ERRORS(cufftCreate(&plan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, mg[i].blockSize, CUFFT_R2C, 1));

		/*Plans*/
		cufftHandle outplan;
		CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, mg[i].blockSize, CUFFT_C2R, 1));

#if defined WIN64 || CB == 0
#else
		/*Create host pointer to CB Function*/
		cufftCallbackLoadC hostCopyOfCallbackPtr;
		checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));

		/*Associate the load callback with the plan*/
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr,
			CUFFT_CB_LD_COMPLEX, (void **)&(mg[i].d_rbuf)));
#endif	

		/*Transform Filter*/
		Print("Transforming filter\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)mg[i].d_rbuf, (cufftComplex*)mg[i].d_rbuf));

		/*Convolving*/
		overlapAdd(mg[i].d_ibuf, (cufftComplex*)mg[i].d_rbuf, mg[i].size, M, mg[i].blockSize, mg[i].numBlocks, plan, outplan);

		CHECK_CUFFT_ERRORS(cufftDestroy(plan));
		CHECK_CUFFT_ERRORS(cufftDestroy(outplan));
	}
	return minmax1;
}
float doubleBlockMethodTwo(passable *p, int numDevs, cudaStream_t *stream, multi_gpu_struct *mg) {
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	/*Allocate memory*/
	Print("Allocating memory\n");
	for (int i = 0; i < numDevs; i++) {
		if(mg[i].size == 0) {
			break;
		}
		cudaSetDevice(i);
		checkCudaErrors(cudaMalloc(&(mg[i].d_ibuf), (mg[i].size + M) * sizeof(float)));
		checkCudaErrors(cudaMalloc(&(mg[i].d_rbuf), (mg[i].blockSize / 2 + 1) * sizeof(cufftComplex)));
	}

	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
	}
	
	fillBuffers(numDevs, stream, false, mg, p);

	/*Finding peak*/
	float minmax1 = 0;
	Print("Finding input peak\n");
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		checkCudaErrors(cudaSetDevice(i));
		float minmax = DExtrema(mg[i].d_ibuf, mg[i].size);
		if (minmax > minmax1)
			minmax1 = minmax;
	}

	/*Convolving*/
	Print("Convolving\n");
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size == 0) break;
		cudaSetDevice(i);
		/*Create cuFFT plan*/
		Print("Creating FFT plans\n");
		cufftHandle plan;
		CHECK_CUFFT_ERRORS(cufftCreate(&plan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, mg[i].blockSize, CUFFT_R2C, 1));

		/*Plans*/
		cufftHandle outplan;
		CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, mg[i].blockSize, CUFFT_C2R, 1));

#if defined WIN64 || CB == 0
#else
		/*Create host pointer to CB Function*/
		cufftCallbackLoadC hostCopyOfCallbackPtr;
		checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr, sizeof(hostCopyOfCallbackPtr)));

		/*Associate the load callback with the plan*/
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void **)&hostCopyOfCallbackPtr,
			CUFFT_CB_LD_COMPLEX, (void **)&(mg[i].d_rbuf)));
#endif	

		/*Transform Filter*/
		Print("Transforming filter\n");
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)mg[i].d_rbuf, (cufftComplex*)mg[i].d_rbuf));

		/*Convolving*/
		overlapAdd(mg[i].d_ibuf, (cufftComplex*)mg[i].d_rbuf, mg[i].size, M, mg[i].blockSize, mg[i].numBlocks, plan, outplan);
		CHECK_CUFFT_ERRORS(cufftDestroy(plan));
		CHECK_CUFFT_ERRORS(cufftDestroy(outplan));

		checkCudaErrors(cudaFree(mg[i].d_rbuf));
	}
	return minmax1;
}
int doubleBlockTest(int numDevs, size_t *freeSizes, multi_gpu_struct *mg, passable *p) {
	long long iFrames = p->input->frames;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;

	Print("Verifying total size of GPUs\n");
	long long totalAllowedFrames = 0;
	int singleDev;
	size_t max = 0;
	for (int i = 0; i < numDevs; i++) {
		/*Theoretically should be 4. Dividing by 8 to be conservative*/
		totalAllowedFrames += freeSizes[i] / 4 / 2;
		totalAllowedFrames -= rFrames;

		/*Finding device with most memory*/
		if (max < totalAllowedFrames) {
			max = totalAllowedFrames;
			singleDev = i;
		}
	}
	/*TODO: Add condition for output memory allocation*/
	if (totalAllowedFrames < iFrames + M * numDevs) {
		Print("totalAllowedFrames < iFrames + M * numDevs failed\n");
		return 0;
		/*
			fprintf(stderr, "\n\nERROR: NOT ENOUGH COLLECTIVE MEMORY ON THE GPUs. EXITING\n\n");
			checkCudaErrors(cudaFreeHost(ibuf));
			free(rbuf);
			return NULL;
		*/
	}

	/*Checking to see if memory will be distributed by method 1 or 2*/
	long long amtPerDevice = (iFrames + numDevs - 1) / numDevs;
	for (int i = 0; i < numDevs; i++) {
		/*Theoretically should be 4, dividing by 8 to be conservative*/
		size_t freeSize = freeSizes[i] / 4 / 2 - rFrames;
		freeSize = pow(2, floor(log2((double)freeSize)));
		if (amtPerDevice + M > freeSize) {
			fprintf(stderr, "Attempting method 2 memory redistribution.\n");
			amtPerDevice = iFrames;
			break;
		}
	}
	if (amtPerDevice == (iFrames + numDevs - 1) / numDevs) return 1;


	/*Filling in data*/
	Print("Filling in data\n");

	/*Making the block size where L = M approx*/
	size_t smallestBlockSize = pow(2, ceil(log2((double)(M * 2))));
	fprintf(stderr, "smallestBlockSize: %lu\n", smallestBlockSize);
	/*Checking to see if there's enough memory*/
	long long framecount = 0;
	for (int i = 0; i < numDevs; i++) {
		if (framecount >= iFrames) {
			mg[i].size = 0;
			mg[i].d_ibuf = NULL;
			mg[i].d_rbuf = NULL;
			mg[i].L = 0;
			mg[i].offset = -1;
			mg[i].numBlocks = 0;
			mg[i].M = 0;
			mg[i].blockSize = 0;
			continue;
		}
		mg[i].M = M;
		mg[i].blockSize = smallestBlockSize;
		mg[i].L = smallestBlockSize - M;

		/*Estimated amount of allocatable floats. Theoretically should be 4, dividing by 8 to be conservative*/
		size_t actualFreeSize = freeSizes[i] / 4U / 2U;

		actualFreeSize -= smallestBlockSize * 2U;
		/*Estimated amount of allocatable floats to the power of 2*/
		size_t inBlockSize = pow(2, ceil(log2((double)actualFreeSize)));
		
		/*Number of elements in input - M*/
		inBlockSize -= M;

		mg[i].offset = framecount;
		mg[i].size = inBlockSize < iFrames - framecount ? inBlockSize : iFrames - framecount;
		mg[i].numBlocks = mg[i].size / mg[i].L;
		framecount += inBlockSize;
		printMGInfo(mg[i]);
	}
	if (framecount >= iFrames) {
		return 2;
	}

	return 0;
}
float *multiGPUFFTDebug(passable *p) {
	setlocale(LC_NUMERIC, "");
	long long iFrames = p->input->frames;
	long long rFrames = p->reverb->frames;
	long long M = rFrames - 1;
	long long oFrames = iFrames + rFrames - 1;
	int streamsPerDev = 4;
	int singleDev = 0;
	float *d_obuf, *obuf;
	float minmax1, minmax2;
	multi_gpu_struct *mg;

	if (p->type != mono_mono) {
		fprintf(stderr, "Option currently unavailable\n");
		return NULL;
	}
	cudaEvent_t start, stop;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	/*Allocate returnable output buffer*/
	checkCudaErrors(cudaMallocHost(&obuf, oFrames * sizeof(float)));

	/*Get number of devices*/
	int numDevs = 0;
	cudaGetDeviceCount(&numDevs);

	cudaStream_t *stream = (cudaStream_t*)malloc(numDevs * streamsPerDev * sizeof(cudaStream_t));

	mg = (multi_gpu_struct*)malloc(numDevs * sizeof(struct multi_gpu_struct));

	/*Get free space in all devices*/
	Print("Get free space in all devices\n");
	size_t *freeSizes = (size_t*)malloc(numDevs * sizeof(size_t));

	
	for (int i = 0; i < numDevs; i++) {
		cudaSetDevice(i);
		freeSizes[i] = getFreeSize();
		printSize();
		for (int j = 0; j < streamsPerDev; j++) {
			checkCudaErrors(cudaStreamCreate(&stream[i * streamsPerDev + j]));
		}
	}

	/*Find out if single block or double block*/
	size_t totalAllowedFrames = 0;
	for (int i = 0; i < numDevs; i++) {
		/*Max number of elements that's a power of 2*/
		size_t freeAmount = pow(2, floor(log2((double)(freeSizes[i] / 4 / 2))));
		totalAllowedFrames += freeAmount - M;
	}

	/*Single Block*/
	if (totalAllowedFrames > iFrames) {
		/*Determining method 1 or method 2*/
		bool method1 = true;
		long long amtPerDevice = (iFrames + numDevs - 1) / numDevs;
		amtPerDevice += M;
		amtPerDevice = pow(2, ceil(log2((double)amtPerDevice)));
		for (int i = 0; i < numDevs; i++) {
			if (freeSizes[i] / 16 / 2 < amtPerDevice) {
				method1 = false;
				break;
			}
		}

		if (method1) {
			Print("SG Block method 1\n");
			/*Do method 1*/
			minmax1 = singleBlockMethodOne(p, numDevs, stream, mg);
		}
		else {
			/*Do method 2*/
			Print("SG Block method 2\n");
			minmax1 = singleBlockMethodTwo(p, numDevs, freeSizes, stream, mg);

		}
	}
	else {
		/*Double Block*/
		int test = doubleBlockTest(numDevs, freeSizes, mg, p);
		if (!test) {
			fprintf(stderr, "\n\nERROR: NOT ENOUGH COLLECTIVE MEMORY ON THE GPUs. EXITING\n\n");
			checkCudaErrors(cudaFreeHost(p->input->buf));
			checkCudaErrors(cudaFreeHost(p->reverb->buf));
			return NULL;

		}
		if (test == 1) {
			Print("Double Block method 1\n");
			minmax1 = doubleBlockMethodOne(p, numDevs, stream, mg);
		}
		else {
			Print("Double Block method 2\n");
			minmax1 = doubleBlockMethodTwo(p, numDevs, stream, mg);
		}
		
	}

	Print("Finding device with largest free space\n");
	/*Find device with largest free space*/
	singleDev = findLargestGPU(numDevs);

	/*Reconstructing */
	d_obuf = overlapAddMultiGPU(mg, p, numDevs, stream, singleDev);

	Print("Cleanup streams\n");
	/*Cleanup streams*/
	for (int i = 0; i < numDevs; i++) {
		checkCudaErrors(cudaSetDevice(i));
		for (int j = 0; j < streamsPerDev; j++) {
			checkCudaErrors(cudaStreamSynchronize(stream[i * 4 + j]));
			checkCudaErrors(cudaStreamDestroy(stream[i * 4 + j]));
		}
		if (mg[i].size == 0) continue;
		checkCudaErrors(cudaFree(mg[i].d_ibuf));
	}

	/*Find output peak*/
	Print("Finding output peak\n");
	checkCudaErrors(cudaSetDevice(singleDev));
	minmax2 = DExtrema(d_obuf, oFrames);
	float scale = minmax1 / minmax2;


	/*Scale & transfer output*/
	Print("Scaling and transferring output\n");
	scaling(obuf, d_obuf, scale, oFrames);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Time for GPU convolution: %f ms\n", milliseconds);
	
	free(mg);
	free(freeSizes);
	free(stream);

	return obuf;
}