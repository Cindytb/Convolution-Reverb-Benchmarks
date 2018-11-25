#include "Audio.cuh"
void errorCheckGPU(int iCh, int iSR, int rSR){
	if(iCh != 1){
        fprintf(stderr, "ERROR: Program can only take mono files\n");
        exit(100);
    }
	if (iSR != rSR) {
		fprintf(stderr, "ERROR: Input and reverb files are two different sample rates\n");
		fprintf(stderr, "Input file is %i\n", iSR);
		fprintf(stderr, "Reverb sample rate is %i\n", rSR);
		exit(200);
	}
}
// Calculates log2 of number.  
long long findSize(long long i_size, long long r_size) {
	int lenY = i_size + 2 * r_size - 2;
	int currPow = 0;
	int lenY2 = pow(2, currPow);

	/* Get first first power of two larger than lenY */
	while (lenY2 < lenY) {
		currPow++;
		lenY2 = pow(2, currPow);
	}
	return lenY2;
}
long long getAudioBlockSize() {
	long long totalGPURAM = getFreeSize();

	/*2 real arrays 8s + 2 complex arrays 8s + 16 =
	16s + 16 --> s = (total - 16) / 16
	Currently dividing by 24 to give some extra room
	*/
	int lenY = totalGPURAM / 24;
	/* Get first first power of two less than lenY */
	return (long long)(pow(2, floor(log2( (double)lenY ) ) ) );
}
void readFileExperimental3(const char *iname, const char *rname, 
	int *iCh, int *iSR, long long *iframes, int *rCh, int *rSR,  long long *rframes, 
	float **d_ibuf, float **d_rbuf, long long *new_size, bool *blockProcessingOn, bool timeDomain) {
	setlocale(LC_NUMERIC, "");
	/*Create cuda streams for concurrent kernels*/
	cudaStream_t streams[2];
	float *ibuf, *rbuf;
	SF_INFO i_info, r_info;
	SNDFILE *i_sndfile, *r_sndfile;
	memset(&i_info, 0, sizeof(i_info));
	memset(&r_info, 0, sizeof(r_info));

	/*Read input*/
	i_sndfile = sf_open(iname, SFM_READ, &i_info);
	if (i_sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", iname);
		exit(1);
	}
	/*Read reverb*/
	r_sndfile = sf_open(rname, SFM_READ, &r_info);
	if (r_sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", rname);
		exit(1);
	}
	/*Store input & reverb metadata*/
	*iSR = i_info.samplerate;
	*iframes = i_info.frames;
	*iCh = i_info.channels;

	*rSR = r_info.samplerate;
	*rframes = r_info.frames;
	*rCh = r_info.channels;
	/*Error check. Terminate program if requirements are not met*/
	errorCheckGPU(*iCh, *iSR, *rSR);
	long long totalSize = *iframes * *iCh;
	int mod = totalSize % 2;
	/*Find padded size for FFT*/
	
	*new_size = pow(2, ceil(log2((double)(totalSize + *rframes * *rCh - 1))));
	if(!timeDomain){
		if(*new_size > getAudioBlockSize()){
			*blockProcessingOn = true;
		}
	}
	int numDevs = 1;
	cudaGetDeviceCount(&numDevs);
	if(*blockProcessingOn && numDevs != 1){
		/*Allocate host pinned memory for input and reverb*/
		checkCudaErrors(cudaMallocHost((void**)&ibuf, totalSize * sizeof(float)));
		rbuf = (float*)malloc( *rframes * *rCh * sizeof(float));
		if (r_info.channels == 1) {
			sf_read_float(r_sndfile, rbuf, *rframes * *rCh);
		}
		else {
			fprintf(stderr, "ERROR: %s : Only mono files allowed", rname);
			exit(100);
		}
		if (i_info.channels == 1) {
			sf_read_float(i_sndfile, ibuf, totalSize);
		}
		else {
			fprintf(stderr, "ERROR: %s : Only mono files allowed", iname);
			exit(100);
		}
		*d_ibuf = ibuf;
		*d_rbuf = rbuf;
		return;
	}
	else if (*blockProcessingOn){
		/*Allocate device memory for input and reverb without*/
		checkCudaErrors(cudaMalloc(d_ibuf, (totalSize + *rframes * *rCh - 1) * sizeof(float)));
		checkCudaErrors(cudaMalloc(d_rbuf, *rframes * *rCh * sizeof(float)));
	}
	else{
		/*Allocate device memory for input and reverb with padding*/
		checkCudaErrors(cudaMalloc(d_ibuf, *new_size * sizeof(float)));
		checkCudaErrors(cudaMalloc(d_rbuf, *new_size * sizeof(float)));
	}
	
	/*Allocate host pinned memory for input and reverb*/
	checkCudaErrors(cudaMallocHost((void**)&ibuf, (totalSize / 2 + mod) * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&rbuf, *rframes * *rCh * sizeof(float)));
	
	/*Read in all reverb audio data*/
	checkCudaErrors(cudaStreamCreate(&streams[0]));
	checkCudaErrors(cudaStreamCreate(&streams[1]));
	if (r_info.channels == 1) {
		sf_read_float(r_sndfile, rbuf, *rframes * *rCh);
	}
	else {
		fprintf(stderr, "ERROR: %s : Only mono files allowed", rname);
        exit(100);
	}
	if(!*blockProcessingOn){
		int numThreads = 512;
		int numBlocks = (*new_size + numThreads - 1) / numThreads;
		FillWithZeros<<<numBlocks, numThreads, 0, streams[0]>>>(*d_rbuf, (*rframes - 1) * *rCh,  *new_size);
		FillWithZeros<<<numBlocks, numThreads, 0, streams[1]>>>(*d_ibuf, totalSize, *new_size);

	}
	
	/*Fill reverb buffer*/
	checkCudaErrors(cudaMemcpyAsync(*d_rbuf, rbuf, *rframes * *rCh * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
	
	/*Read in all input audio data*/
	if (i_info.channels == 1) {
	 	sf_read_float(i_sndfile, ibuf, totalSize / 2);
	 	checkCudaErrors(cudaMemcpy(*d_ibuf, ibuf, totalSize / 2 * sizeof(float), cudaMemcpyHostToDevice));
	 	sf_read_float(i_sndfile, ibuf, totalSize / 2 + mod);
	 	checkCudaErrors(cudaMemcpyAsync(*d_ibuf + totalSize / 2, ibuf, (totalSize / 2 + mod) * sizeof(float), cudaMemcpyHostToDevice, streams[1]));	
	}
	else {
		fprintf(stderr, "ERROR: %s : Only mono files allowed", iname);
        exit(100);
	}
	checkCudaErrors(cudaFreeHost(rbuf));
	sf_close(i_sndfile);
	sf_close(r_sndfile);
	checkCudaErrors(cudaStreamSynchronize(streams[0]));
	checkCudaErrors(cudaStreamDestroy(streams[0]));
	checkCudaErrors(cudaStreamSynchronize(streams[1]));
	checkCudaErrors(cudaStreamDestroy(streams[1]));
	
	checkCudaErrors(cudaFreeHost(ibuf));
	
}
/*Write file with variable SR*/

void writeFile(const char * name, float * buf, long long size, int fs, int ch) {
	int format = 0;
	if(size < 1073741824){
		format |= SF_FORMAT_WAV;
	}
	else{
		format |= SF_FORMAT_W64;
	}
	
	format |=  SF_FORMAT_FLOAT;
	SndfileHandle file = SndfileHandle(name, SFM_WRITE, format, ch, fs);
	file.writef(buf, size);
} 
