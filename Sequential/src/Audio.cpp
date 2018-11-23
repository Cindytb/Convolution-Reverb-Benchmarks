#include "Audio.h"
void errorCheck(int iCh, int iSR, int rSR){
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
/*Input name, address to buffer to be written, and address of number of channels. Output size.*/
long long readFile(const char *name, float **buf, int *numCh, int *SR) {
	SF_INFO info; 
	SNDFILE *sndfile; 
	memset(&info, 0, sizeof(info)); 
	info.format = 0; 
	sndfile = sf_open(name, SFM_READ, &info); 
	if (sndfile == NULL) { 
		fprintf(stderr, "ERROR. Cannot open %s\n", name); 
		exit(1); 
	} 
	sf_count_t size = info.frames; 
	*numCh = info.channels; 
	*SR = info.samplerate;
	*buf = (float*) malloc(sizeof(float) * size * info.channels); 
	if(*buf == 0){ 
		fprintf(stderr, "ERROR: Cannot allocate enough memory\n"); 
		exit(2); 
	} 
	
	if (info.channels < 3) { 
		sf_read_float(sndfile, *buf, size * info.channels); 
	} 
	else {
		fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", name);
	}

	size *= *numCh;
	sf_close(sndfile);
	return size;
}
void readFileExperimental(const char *iname, const char *rname, 
	int *iCh, int *iSR, long long *iframes, int *rCh, int *rSR,  long long *rframes, 
	float **ibuf, float **rbuf, long long *new_size, bool *blockProcessingOn, bool timeDomain) {
	float *buf;
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
	errorCheck(*iCh, *iSR, *rSR);
	long long totalSize = *iframes * *iCh;

	/*Find padded size for FFT*/
	*new_size = pow(2, ceil(log2((double)(totalSize + *rframes * *rCh - 1))));
	if( timeDomain || (*ibuf = fftwf_alloc_real(*new_size * 2)) ){
		*blockProcessingOn = true; 
		*ibuf = (float*) malloc(totalSize * sizeof(float)); 
		if(*ibuf == 0){ 
			fprintf(stderr, "ERROR: Cannot allocate enough memory\n"); 
			exit(2); 
		} 
	
		if (*iCh < 3) { 
			sf_read_float(i_sndfile, *ibuf, totalSize); 
		} 
		else {
			fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", iname);
		}
		*rbuf = (float*) malloc( (*rframes * *rCh) * sizeof(float));
		if(*rbuf == 0){ 
			fprintf(stderr, "ERROR: Cannot allocate enough memory\n"); 
			exit(2); 
		} 
	
		if (*rCh < 3) { 
			sf_read_float(r_sndfile, *rbuf, *rframes * *rCh); 
		} 
		else {
			fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", rname);
		}
		sf_close(i_sndfile);
		sf_close(r_sndfile);
		return;
	}
	*blockProcessingOn = false;
	*rbuf = *ibuf + *new_size;
	if (*iCh < 3) { 
		sf_read_float(i_sndfile, *ibuf, totalSize); 
	} 
	else {
		fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", iname);
	}
	if (*rCh < 3) { 
			sf_read_float(r_sndfile, *rbuf, *rframes * *rCh); 
	} 
	else {
		fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", rname);
	}
	
	/*Pad remaining values to 0*/
    for(long long i = *rframes * *rCh; i < *new_size; i++){
        (*rbuf)[i] = 0.0f;
        if(i >= totalSize){
            (*ibuf)[i] = 0.0f;
        }
    }
	sf_close(i_sndfile);
	sf_close(r_sndfile);
}
/*Write file with 44.1k SR*/
/*
void writeFile(const char * name, float * buf, long long size, int ch) {
	int format = 0;
	format |= SF_FORMAT_WAV;
	format |= SF_FORMAT_PCM_24;
	SndfileHandle file = SndfileHandle(name, SFM_WRITE, format, ch, 44100);
	file.writef(buf, size);

}
*/
/*Write file with variable SR*/

void cpuWriteFile(const char * name, float * buf, long long size, int fs, int ch) {
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
