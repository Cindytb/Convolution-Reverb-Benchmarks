#include "Audio.h"

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
