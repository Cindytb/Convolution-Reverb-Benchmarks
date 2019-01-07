#pragma once 
#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H

enum flags {
	mono_mono,
	mono_stereo,
	stereo_mono,
	stereo_stereo,
};

typedef struct audio_container {
	long long frames;
	float *d_buf;
	float *buf;
	int channels;
} audio_container;

typedef struct multi_gpu_struct {
	float *d_ibuf;
	float *d_rbuf;
	long long size;
	long long L;
	long long offset;
	size_t blockSize;
	int numBlocks;
	int M;
} multi_gpu_struct;

typedef struct passable {
	audio_container *input;
	audio_container *reverb;
	multi_gpu_struct *mg_struct;
	long long paddedSize;
	enum flags type;
} passable;

#define CB 1
//CB = 0 -- force no callback
//CB = 1 -- force use callback unless WIN64 is defined

#define CONCURRENTKERNELCPY 1
//CONCURRENTKERNELCPY = 0 Not concurrent. Separate scale and copy functions
//CONCURRENTKERNELCPY = 1 Scale kernel then copy, 4 streams
//CONCURRENTKERNELCPY = 2 Scale kernel 4x then copy 4x

#define _DEBUG 1
#ifdef _DEBUG
#define Print(s)  fprintf(stderr, "%s", s)
#else
#define Print(s)  
#endif

#endif