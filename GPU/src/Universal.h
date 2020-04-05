#pragma once 
#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_

/*Using C++ style enums and structs*/
enum flags {
	mono_mono,
	mono_stereo,
	stereo_mono,
	stereo_stereo,
};

struct audio_container {
	size_t frames;
	float *d_buf;
	float *buf;
	int channels;
};

struct db_block_struct{
	int L;
	int numBlocks;
};

struct multi_gpu_struct {
	float *d_ibuf;
	float *d_rbuf;
	size_t size;
	size_t blockSize;
	long long offset;
	long L;
	int numBlocks;
	int M;
};

struct passable {
	audio_container *input;
	audio_container *reverb;
	multi_gpu_struct *mg_struct;
	long long paddedSize;
	flags type;
};

#define CB 1
//CB = 0 -- force no callback
//CB = 1 -- force use callback unless WIN64 is defined

#define CONCURRENTKERNELCPY 1
//CONCURRENTKERNELCPY = 0 Not concurrent. Separate scale and copy functions
//CONCURRENTKERNELCPY = 1 Scale kernel then copy, 4 streams
//CONCURRENTKERNELCPY = 2 Scale kernel 4x then copy 4x

#ifdef _DEBUG
#define Print(s)  fprintf(stderr, "%s", s)
#else
#define Print(s)  
#endif

#endif