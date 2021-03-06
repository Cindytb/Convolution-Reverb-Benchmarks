#include "Convolution.cuh"

extern __device__ cufftCallbackLoadC myOwnCallbackPtr;
void printMGInfo(multi_gpu_struct m){
	fprintf(stderr, "d_ibuf    : %p\n", m.d_ibuf);
	fprintf(stderr, "d_rbuf    : %p\n", m.d_rbuf);
	fprintf(stderr, "size      : %'lu\n", m.size);
	fprintf(stderr, "blockSize : %'lu\n", m.blockSize);
	fprintf(stderr, "offset    : %'lli\n", m.offset);
	fprintf(stderr, "L         : %'li\n", m.L);
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

/*Fills in how the signal is divided to each GPU by method 1*/
void fillingInDataSG1(multi_gpu_struct *mg, size_t iFrames, int numDevs, size_t rFrames){
	/*Filling in data*/
	Print("Filling in data\n");
	size_t inChunk = (iFrames + numDevs - 1) / numDevs;
	size_t blockSize = (size_t) pow(2, ceil(log2((double)(inChunk + rFrames - 1))));
	for (int i = 0; i < numDevs; i++) {
		mg[i].offset = inChunk * i;
		mg[i].blockSize = blockSize;
		mg[i].numBlocks = 0;
		mg[i].M = rFrames - 1;
		if (i == numDevs - 1) {
			mg[i].L = iFrames - mg[i].offset;
			mg[i].size = mg[i].L;
		}
		else {
			mg[i].L = inChunk;
			mg[i].size = mg[i].L;
		}
	}
}
/*Fills in how the signal is divided to each GPU by method 2*/
void fillingInDataSG2(multi_gpu_struct *mg, size_t *freeSizes, long long iFrames, int numDevs, long long M){
	/*Filling in data*/
	Print("Filling in data\n");
	size_t framecount = 0;
	for (int i = 0; i < numDevs; i++) {

		mg[i].M = (int) M;
		mg[i].numBlocks = 0;
		mg[i].L = 0;
		if (framecount == (size_t) iFrames) {
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
		mg[i].size = chunk < ( (size_t) iFrames - framecount) ? chunk : (size_t) iFrames - framecount;
		mg[i].blockSize = pow(2, ceil(log2((double)mg[i].size)));
		framecount += mg[i].size;
	}
}
float singleBlock(passable *p, int numDevs, size_t *freeSizes, cudaStream_t *stream, multi_gpu_struct *mg, int sg) {
	size_t iFrames = p->input->frames;
	size_t rFrames = p->reverb->frames;

	if(sg == 1){
		fillingInDataSG1(mg, iFrames, numDevs, rFrames);
	}
	else{
		fillingInDataSG2(mg, freeSizes, iFrames, numDevs, rFrames - 1);
	}

	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
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
	#ifdef openMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size != 0){
			cudaSetDevice(i);
			convolveBatched(mg[i].d_ibuf, mg[i].blockSize);
		}
	}

	return minmax1;
}
float doubleBlockConvolve(multi_gpu_struct *mg, int numDevs, cudaStream_t *stream, passable *p){
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
	#ifdef openMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < numDevs; i++) {
		if (mg[i].size != 0) {
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
			overlapAdd(mg[i].d_ibuf, (cufftComplex*)mg[i].d_rbuf, mg[i].size, mg[i].M, mg[i].blockSize, mg[i].numBlocks, plan, outplan);
			CHECK_CUFFT_ERRORS(cufftDestroy(plan));
			CHECK_CUFFT_ERRORS(cufftDestroy(outplan));

			checkCudaErrors(cudaFree(mg[i].d_rbuf));
		}
	}
	return minmax1;
}
float doubleBlockMethodOne(passable *p, int numDevs, cudaStream_t *stream, multi_gpu_struct *mg) {
	size_t iFrames = p->input->frames;
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
	
	return doubleBlockConvolve(mg, numDevs, stream, p);
}
float doubleBlockMethodTwo(passable *p, int numDevs, cudaStream_t *stream, multi_gpu_struct *mg) {
	/*Allocate memory*/
	Print("Allocating memory\n");
	#ifdef openMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < numDevs; i++) {
		if(mg[i].size == 0) {
			mg[i].blockSize = 0;
		}
		else{
			cudaSetDevice(i);
			checkCudaErrors(cudaMalloc(&(mg[i].d_ibuf), (mg[i].size + mg[i].M) * sizeof(float)));
			checkCudaErrors(cudaMalloc(&(mg[i].d_rbuf), (mg[i].blockSize / 2 + 1) * sizeof(cufftComplex)));
		}
		
	}

	for(int i = 0; i < numDevs; i++){
		printMGInfo(mg[i]);
	}
	return doubleBlockConvolve(mg, numDevs, stream, p);
}
bool checkDistribution(multi_gpu_struct *mg, int numDevs, size_t iFrames, size_t *freeSizes){
	size_t framecount = 0;
	for(int i = 0; i < numDevs; i++){
		if(framecount >=  iFrames){
			
			mg[i].d_ibuf = NULL;
			mg[i].d_rbuf = NULL;
			mg[i].size = 0;

			mg[i].offset = -1;
			mg[i].L = 0;
			mg[i].numBlocks = 0;
			continue;
		}
		size_t actualFreeSize = freeSizes[i] / 4U / 2U ;
		actualFreeSize -= mg[i].blockSize * 2U;
		/*Estimated amount of allocatable floats to the power of 2*/
		size_t inBlockSize = pow(2, ceil(log2((double)actualFreeSize)));
		
		/*Number of elements in input - M*/
		inBlockSize -= mg[i].M;

		mg[i].offset = (long long) framecount;
		mg[i].size = inBlockSize < iFrames - framecount ? inBlockSize : iFrames - framecount;
		mg[i].L = mg[i].blockSize - mg[i].M;
		mg[i].numBlocks = mg[i].size / mg[i].L;
		framecount += mg[i].size;
	}
	if(framecount < iFrames){
		return false;
	}
	for(int i = 0; i < numDevs; i++){
		if(mg[i].size == 0)
			break;
		checkCudaErrors(cudaSetDevice(i));
		size_t workspace;
		CHECK_CUFFT_ERRORS(cufftEstimate1d(mg[i].blockSize, CUFFT_R2C, 2, &workspace));
		if(freeSizes[i] < workspace + (mg[i].blockSize / 2 + 1) * 8UL * 2UL){
			return false;
		}
		
	}
	return true;

}
int doubleBlockTest(int numDevs, size_t *freeSizes, multi_gpu_struct *mg, passable *p) {
	size_t iFrames = p->input->frames;
	size_t rFrames = p->reverb->frames;
	size_t M = rFrames - 1;

	Print("Verifying total size of GPUs\n");
	size_t totalAllowedFrames = 0;
	//int singleDev;
	size_t max = 0;
	for (int i = 0; i < numDevs; i++) {
		/*Theoretically should be 4. Dividing by 8 to be conservative*/
		totalAllowedFrames += freeSizes[i] / 4 / 2 / 2;
		totalAllowedFrames -= rFrames;

		/*Finding device with most memory*/
		if (max < totalAllowedFrames) {
			max = totalAllowedFrames;
			//singleDev = i;
		}
	}
	/*TODO: Add condition for output memory allocation*/
	if (totalAllowedFrames < (size_t) (iFrames + M * numDevs)) {
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
	size_t amtPerDevice = (iFrames + numDevs - 1) / numDevs;
	for (int i = 0; i < numDevs; i++) {
		/*Theoretically should be 4, dividing by 8 to be conservative*/
		size_t freeSize = freeSizes[i] / 4 / 2 / 4 - rFrames;
		freeSize = pow(2, floor(log2((double)freeSize)));
		if (amtPerDevice + (size_t) M > freeSize) {
			fprintf(stderr, "Attempting method 2 memory redistribution.\n");
			amtPerDevice = iFrames;
			break;
		}
	}
	if (amtPerDevice == (size_t) (iFrames + numDevs - 1) / numDevs) return 1;


	/*Filling in data*/
	Print("Filling in data\n");

	/*Making the block size where L = M approx*/
	size_t smallestBlockSize = pow(2, ceil(log2((double)(M * 2))));
	fprintf(stderr, "smallestBlockSize: %'lu\n", smallestBlockSize);

	/*Checking to see if there's enough memory*/
	for (int i = 0; i < numDevs; i++) {
		mg[i].M = M;
		mg[i].blockSize = smallestBlockSize;
		printMGInfo(mg[i]);
	}
	if (!checkDistribution(mg, numDevs, iFrames, freeSizes)) {
		return 0;
	}

	/*Increasing the block size on each device to find the optimal setup
		where block size on each device is maximized*/
	int fullCount = 0;
	int dev = 0;
	//int merp = 0;

	Print("Attempting redistribution optimization\n");
	/*TODO: Needs thorough testing*/
	do{
		if(mg[dev].numBlocks){
			if(mg[dev].offset != -1){
				mg[dev].blockSize *= 2;
			}
			if(!checkDistribution(mg, numDevs, iFrames, freeSizes)){
				mg[dev].blockSize /= 2;
				checkDistribution(mg, numDevs, iFrames, freeSizes);
				fullCount++;
			}
			else{
				fullCount = 0;
			}
		}
		else{
			fullCount++;
		}
		// fprintf(stderr, "\n\nIteration: %i\n", merp++);
		// for(int i = 0; i < numDevs; i++){
		// 	printMGInfo(mg[i]);
		// }
		dev++;
		dev %= numDevs;
	}
	while(fullCount < numDevs);
	return 2;
}
float *multiGPUFFTDebug(passable *p) {
	setlocale(LC_NUMERIC, "");
	size_t iFrames = p->input->frames;
	size_t rFrames = p->reverb->frames;
	size_t oFrames = iFrames + rFrames - 1;
	int streamsPerDev = 4;
	int singleDev = 0;
	float *d_obuf, *obuf;
	float minmax1, minmax2;
	multi_gpu_struct *mg;

	if (p->type != mono_mono) {
		fprintf(stderr, "Option currently unavailable\n");
		return NULL;
	}
	
	/*Create and start a timer*/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	/*Allocate returnable output buffer*/
	checkCudaErrors(cudaMallocHost(&obuf, oFrames * sizeof(float)));

	/*Get number of devices*/
	int numDevs = 0;
	cudaGetDeviceCount(&numDevs);

	/*Allocate memory for streams & passable structs*/
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
		size_t freeAmount = pow(2, floor(log2((double)(freeSizes[i] / 4U / 4U))));
		totalAllowedFrames += (freeAmount / 2);
	}

	/*Single Block*/
	if (totalAllowedFrames > (size_t)iFrames) {
		/*Determining method 1 or method 2*/
		bool method1 = true;
		size_t amtPerDevice = (iFrames + numDevs - 1) / numDevs;
		amtPerDevice = pow(2, ceil(log2((double)amtPerDevice) + 1));
		fprintf(stderr, "amtPerDevice: %lu\n", amtPerDevice);
		for (int i = 0; i < numDevs; i++) {
			if (freeSizes[i] / 4U / 2U < amtPerDevice) {
				method1 = false;
				Print("SG Block Method 2\n");
				break;
			}
		}
		int method = method1 ? 1 : 2;
		if(method1) Print("SG Block Method 1\n");
		minmax1 = singleBlock(p, numDevs, freeSizes, stream, mg, method);
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