#include "Audio.cuh"
/*Debugging print function*/
void printArr(float *buf, int len){
	for(int i = 0; i < len - 1; i++){
		printf("%2.0f ", buf[i]);
	}
	printf("%2.0f\n", buf[len - 1]);
}
__global__ void deviceDeInterleave(float *d_buf, long long size){
	deInterleave(d_buf, size);
}
void printType(passable *p){
	switch(p->type){
		case 0:
			Print("Flag Type: mono_mono\n");
			break;
		case 1:
			Print("Flag Type: mono_stereo\n");
			break;
		case 2:
			Print("Flag Type: stereo_mono\n");
			break;
		case 3:
			Print("Flag Type: stereo_stereo\n");
			break;
	}
}
void errorCheckGPU(int iCh, int rCh, int iSR, int rSR, passable *p){
	if(iCh == 1 && rCh == 1){
		p->type = mono_mono;
	}
	else if(iCh == 1 && rCh == 2){
		p->type = mono_stereo;
	}
	else if(iCh == 2 && rCh == 1){
		p->type = stereo_mono;
	}
	else if(iCh == 2 && rCh == 2){
		p->type = stereo_stereo;
	}
	else{
		fprintf(stderr, "ERROR: Current channel setup not supported. Only mono/stereo files allowed.\n");
		exit(100);
	}
	if (iSR != rSR) {
		fprintf(stderr, "ERROR: Input and reverb files are two different sample rates\n");
		fprintf(stderr, "Input file is %i\n", iSR);
		fprintf(stderr, "Reverb sample rate is %i\n", rSR);
		exit(200);
	}
}
long long getAudioBlockSize(enum flags flag) {
	long long totalGPURAM = getFreeSize();
	long long lenY;
	if(flag == mono_mono){
		/*
			2 complex arrays --> 2 * 8(s / 2 + 1) =
			2(4s + 8) =
			8s + 16 = 
			s = (total - 16) / 8
			Currently dividing by 16 to give some extra room
		*/
		lenY = totalGPURAM / 16 ;
	}
	else if(flag == mono_stereo || flag == stereo_mono){
		/*
			3 complex arrays --> 3 * 8(s / 2 + 1)
			3(4s + 8) = 
			12s + 24 =
			s = (total - 24) / 12
			Currently dividing by 24 to give extra room
		*/
		lenY = totalGPURAM / 24;
	}
	else if(flag == stereo_stereo){
		/*
			4 complex arrays --> 4 * 8(s / 2 + 1)
			4(4s + 8) = 
			16s + 32 =
			s = (total - 32) / 16
			Currently dividing by 32 to give extra room
		*/
		lenY = totalGPURAM / 32;
	}
	else{
		fprintf(stderr, "ERROR: Channel mode not supported. Exiting program\n");
		exit(100);
	}
	/* Get first first power of two less than lenY */
	return (long long)(pow(2, floor(log2( (double)lenY ) ) ) );
}
void readFileExperimentalDebug(const char *iname, const char *rname, 
	int *SR, bool *blockProcessingOn, bool timeDomain, passable *p) {
	setlocale(LC_NUMERIC, "");
	int iSR, rSR;
	long long totalSize;
	cudaStream_t streams[8];
	float *ibuf, *rbuf;
	float *d_ibuf, *d_rbuf;
	long long rFrames, iFrames;
	int rCh, iCh;
	SF_INFO i_info, r_info;
	SNDFILE *i_sndfile, *r_sndfile;
	int numThreads = 512;
	int numBlocks;

	memset(&i_info, 0, sizeof(i_info));
	memset(&r_info, 0, sizeof(r_info));

	/*Open input*/
	i_sndfile = sf_open(iname, SFM_READ, &i_info);
	if (i_sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", iname);
		exit(1);
	}
	/*Open reverb*/
	r_sndfile = sf_open(rname, SFM_READ, &r_info);
	if (r_sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", rname);
		exit(1);
	}
	
	/*Store input & reverb metadata*/
	iSR = i_info.samplerate;
	iFrames = i_info.frames;
	p->input->frames = iFrames;
	iCh = i_info.channels;
	p->input->channels = iCh;

	rSR = r_info.samplerate;
	rFrames = r_info.frames;
	p->reverb->frames = rFrames;
	rCh = r_info.channels;
	p->reverb->channels = rCh;

	Print("Error checking\n");
	/*Error check. Terminate program if requirements are not met*/
	/*Fill flag option*/
	errorCheckGPU(iCh, rCh, iSR, rSR, p);
	*SR = iSR;
	totalSize = iFrames * iCh;

	printType(p);

	if(!timeDomain){
		/*Find padded size for FFT*/
		Print("Finding padded size for FFT\n");
		p->paddedSize = pow(2, ceil(log2((double)(iFrames + rFrames - 1))));
	
		Print("Checking to see if block processing is necessary\n");
		/*Check memory to see if input needs to be processed in chunks*/
		//if(p->paddedSize > getAudioBlockSize(p->type)){
			*blockProcessingOn = true;
		//}
	}
	else{
		p->paddedSize = iFrames + rFrames - 1;
	}
	int oCh = 1;
	if(p->type != mono_mono){
		oCh = 2;
	}
	/*Allocating Memory!*/
	int numDevs = 1;
	cudaGetDeviceCount(&numDevs);

	/*Multiple Device Block Processing*/
	if(*blockProcessingOn && numDevs != 1){
		Print("Allocating Device Memory For Multiple Device Block Processing\n");
		/*Allocate host pinned memory for input and reverb*/
		checkCudaErrors(cudaMallocHost((void**)&ibuf, iFrames * iCh * sizeof(float)));
		checkCudaErrors(cudaMallocHost((void**)&rbuf, rFrames * rCh * sizeof(float)));
		Print("Reading in input\n");
		sf_read_float(r_sndfile, rbuf, rFrames * rCh);
		
		/*De-interleave input if stereo*/
		if(iCh == 2){
			Print("De-interleaving input\n");
			deInterleave(ibuf, iFrames * iCh);
		}
		Print("Reading in filter\n");
		sf_read_float(i_sndfile, ibuf, iFrames * iCh);
		
		/*De-interleave reverb if stereo*/
		if(rCh == 2){
			Print("de-interleaving reverb\n");
			deInterleave(rbuf, rFrames * rCh);
		}
		p->input->buf = ibuf;
		p->reverb->buf = rbuf;
		return;
	}
	/*Single Device Block*/
	else if (*blockProcessingOn){
		/*Allocate device memory for input but not reverb*/
		Print("Allocating Device Memory For Single Device Block Processing\n");
		checkCudaErrors(cudaMalloc(&(p->input->d_buf), (iFrames + rFrames - 1) * oCh * sizeof(float)));
		
	}
	/*Time Domain Block Processing*/
	else if(timeDomain){
		Print("Allocating Device Memory For Time Domain\n");
		/*Allocate device memory for input and reverb with NO padding*/
		checkCudaErrors(cudaMalloc(&(p->input->d_buf), iFrames * iCh * sizeof(float)));
		checkCudaErrors(cudaMalloc(&(p->reverb->d_buf), rFrames * rCh * sizeof(float)));
	}
	/*Single Device Regular*/
	else{
		Print("Allocating Device Memory For Regular\n");
		/*Allocate device memory for input and reverb with padding*/
		checkCudaErrors(cudaMalloc(&(p->input->d_buf), 
			(p->paddedSize * iCh + iCh + p->paddedSize * rCh + rCh) * sizeof(cufftComplex)));
			
		p->reverb->d_buf = p->input->d_buf + p->paddedSize * iCh + iCh;
	}
	d_ibuf = p->input->d_buf;
	d_rbuf = p->reverb->d_buf;
	printMe(p);
	

	/*Allocate host pinned memory for input and reverb*/
	checkCudaErrors(cudaMallocHost((void**)&rbuf, rFrames * rCh * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void**)&ibuf, totalSize * sizeof(float)));
	p->input->buf = ibuf;
	p->reverb->buf = rbuf;
	/*Create Streams*/
	for(int i = 0; i < sizeof(streams)/sizeof(cudaStream_t); i++){
	 	checkCudaErrors(cudaStreamCreate(&streams[i]));
	}

	
	long long iend = p->paddedSize + 2;
	long long rend = p->paddedSize + 2;
	if(timeDomain){
		iend = iFrames;
		rend = rFrames;
	}
	else if(*blockProcessingOn){
		rend = rFrames;
		iend = iFrames + rFrames - 1;
	}
	/*Do padding*/
	if(!*blockProcessingOn && !timeDomain){
		Print("Padding reverb\n");
		/*Pad reverb*/
		numBlocks = (p->paddedSize - rFrames - 1) / numThreads;
		FillWithZeros<<<numBlocks, numThreads, 0, streams[0]>>>(d_rbuf, rFrames, p->paddedSize + 2);
		
		
		/*Pad the reverb if stereo*/
		if(rCh == 2){
			numBlocks = (p->paddedSize - rFrames - 1) / numThreads;
			FillWithZeros<<<numBlocks, numThreads, 0, streams[3]>>>(d_rbuf, p->paddedSize + 2 + rFrames, (p->paddedSize + 2)* 2 );
		}
		
	}
	if(!timeDomain){
		Print("Padding input\n");
		/*Pad input*/
		numBlocks = (iend - iFrames - 1) / numThreads;
		FillWithZeros<<<numBlocks, numThreads, 0, streams[1]>>>(d_ibuf, iFrames, iend);
		
		/*Pad input if stereo*/
		if(iCh == 2){
			numBlocks = (iend - iFrames - 1) / numThreads;
			FillWithZeros<<<numBlocks, numThreads, 0, streams[2]>>>(d_ibuf, iend + iFrames, (iend)* 2);
		}
	}
	
	/*Reading in input*/
	Print("Reading in input\n");
	if(sf_read_float(i_sndfile, ibuf, totalSize) != totalSize){
		Print("ERROR: Reading input file failed\n");
	}

	/*De-interleave input if stereo*/
	if(iCh == 2){
		Print("De-interleaving input\n");
		deInterleave(ibuf, iFrames * iCh);
	}

	/*Copy input to device*/
	checkCudaErrors(cudaMemcpyAsync(d_ibuf, ibuf, iFrames * sizeof(float), 
		cudaMemcpyHostToDevice, streams[4]));
	if(iCh == 2){
		checkCudaErrors(cudaMemcpyAsync(d_ibuf + iend, ibuf + iFrames, iFrames * sizeof(float), 
			cudaMemcpyHostToDevice, streams[5]));
	}

	Print("Reading in reverb\n");
	/*Read in all reverb audio data*/
	if (sf_read_float(r_sndfile, rbuf, rFrames * rCh) != rFrames * rCh)
		fprintf(stderr, "ERROR: Reading reverb file failed\n");

	/*De-interleave reverb if stereo*/
	if(rCh == 2){
		Print("de-interleaving reverb\n");
		deInterleave(rbuf, rFrames * rCh);
	}

	if(!*blockProcessingOn){
		/*Copy reverb to device*/
		checkCudaErrors(cudaMemcpyAsync(d_rbuf, rbuf, rFrames * sizeof(float), 
			cudaMemcpyHostToDevice, streams[6]));
		if(p->type == mono_stereo || p->type == stereo_stereo){
			checkCudaErrors(cudaMemcpyAsync(d_rbuf + rend, rbuf + rFrames, rFrames * sizeof(float), 
				cudaMemcpyHostToDevice, streams[7]));
		}
	}
	
	for(int i = 0; i < sizeof(streams)/sizeof(cudaStream_t); i++){
		checkCudaErrors(cudaStreamSynchronize(streams[i]));
		checkCudaErrors(cudaStreamDestroy(streams[i]));
	}	
	
	sf_close(r_sndfile);
	sf_close(i_sndfile); 
	checkCudaErrors(cudaFreeHost(ibuf));
	if(!*blockProcessingOn){
		checkCudaErrors(cudaFreeHost(rbuf));
	}
	
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

/*Function to swap two elements*/
__device__ __host__  void swap (float *a, float *b){
	float temp = *a;
	*a = *b;
	*b = temp;
}
/*Function to reverse an array*/
__device__ __host__ void reverse (float *arr, int low, int high){
	while( low < high){		
		swap (&arr[low], &arr[high]);
		low++;
		high--;
	}
}
// Cycle leader algorithm to move all even positioned elements 
// at the end. 
__device__ __host__ void cycleLeader( float*buf, int shift, int len){
	int j; 
	float item; 
	for (int i = 1; i < len; i *= 3 ) { 
		j = i; 
		item = buf[j + shift]; 
		do{ 
			// odd index 
			if ( j & 1 ) {
				j = len / 2 + j / 2;
			}
			// even index 
			else{
				j /= 2; 
			}
			// keep the back-up of element at new position 
			swap (&buf[j + shift], &item); 
		} 
		while ( j != i ); 
	} 
}


__device__ __host__ void deInterleave(float *buf, long long samples){
	long long lenRemaining = samples, shift = 0;
	while(lenRemaining){
		int k = floor(log10f(lenRemaining - 1) / log10f(3));
		int currLen = (int)(powf(3, k)) + 1;
		lenRemaining -= currLen;
		cycleLeader(buf, shift, currLen);
		reverse(buf, shift / 2, shift - 1);
		reverse(buf, shift, shift + currLen/2 - 1);
		reverse(buf, shift / 2, shift + currLen/2 - 1);
		shift += currLen;
		
	}
}
/*BROKEN FOR SPECIFIC INPUTS!!!!!!*/
void interleave(float *buf, long long frames){
	long long start = frames, j = frames, i = 0;
	for(long long done = 0; done < 2 * frames - 2; done++){
		if (start == j){
			start--;
			j--;
		}
		i = j > frames - 1 ? j - frames: j;
		j = j > frames - 1 ? 2 * i + 1 : 2 * i;	
		swap(&buf[start], &buf[j]);
	}
}

