#include "main.h"

float* seqEntry(std::string input, std::string reverb, std::string out, bool timeDomain){
	
	float *ibuf, *rbuf, *obuf;
	long long i_size = 0, r_size = 0, o_size = 0;
	int iCh, iSR, rCh, rSR;

	i_size = readFile(input.c_str(), &ibuf, &iCh, &iSR);

	r_size = readFile(reverb.c_str(), &rbuf, &rCh, &rSR);

	if (iSR != rSR) {
		fprintf(stderr, "ERROR: Input and reverb files are two different sample rates\n");
		fprintf(stderr, "Input file is %i\n", iSR);
		fprintf(stderr, "Reverb sample rate is %i\n", rSR);
		exit(2);
	}
	int oCh = iCh == 2 || rCh == 2 ? 2 : 1;
	o_size = i_size / iCh + r_size / rCh - 1;
	long long smallerFrames = r_size / rCh < i_size / iCh ? r_size / rCh  : i_size / iCh;
	long long biggerFrames = r_size / rCh > i_size / iCh ? r_size / rCh  : i_size / iCh;
	
	float *biggerBuf, *smallerBuf;
	int firstCh, secondCh;
	if(biggerFrames == r_size){
			biggerBuf = rbuf;
			smallerBuf = ibuf;
			firstCh = rCh;
			secondCh = iCh;
	}
	else{
		biggerBuf = ibuf;
		smallerBuf = rbuf;
		firstCh = iCh;
		secondCh = rCh;
	}


	std::clock_t c_start = std::clock();
	if(!timeDomain){	
		obuf = entry(ibuf, rbuf, i_size / iCh, r_size / rCh, iCh, rCh);
	}
	else{
		obuf = (float*)malloc(o_size * oCh * sizeof(float));
		TDconvolution(biggerBuf, smallerBuf, biggerFrames, smallerFrames, firstCh, secondCh, obuf);
		maxScale(ibuf, i_size, o_size * oCh, obuf);
	}
	
	std::clock_t c_end = std::clock();
	double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
	fprintf(stderr, "Time for CPU convolution: %f ms\n", time_elapsed_ms);
	if (out.c_str()[0] != ' '){
		fprintf(stderr, "Writing output file %s\n", out.c_str());
		cpuWriteFile(out.c_str(), obuf, o_size * oCh, iSR, iCh);
	}
	
	return obuf;
}

void TDconvolution(float *ibuf, float *rbuf, size_t iframes, size_t rframes, int iCh, int rCh, float *obuf){
	int oframes = iframes + rframes - 1;

	/*Mono and Mono*/
	if(iCh == 1 && rCh == 1){
		for(int i = 0; i < oframes; i++){
			obuf[i] = 0.0f;
		}
		for(size_t n = 0; n < iframes; n++){
			for(size_t j = 0; j < rframes; j++){
				obuf[j + n] += ibuf[n] * rbuf[j];
			}
		}
	}
	
	/*Stereo and Stereo*/
	else if(iCh == rCh){
		for(int i = 0; i < 2; i++){
			for(size_t j = 0; j < oframes; j++){
				obuf[j * 2 + i] = 0;
				for(size_t k = 0; k < rframes; k++){
					if( (j - k) >= 0 && (j - k) <= iframes){
						obuf[j * 2 + i] += ibuf[(j - k) * 2 + i] * rbuf[k * 2 + i];
					}
				}
			}
		}
	}
	/*Stereo and Mono*/
	else if(iCh == 2){
		for(int i = 0; i < 2; i++){
			for(size_t j = 0; j < oframes; j++){
				obuf[j * 2 + i] = 0;
				for(size_t k = 0; k < rframes; k++){
					if( (j - k) >= 0 && (j - k) <= iframes){
						obuf[j * 2 + i] += ibuf[(j - k) * 2 + i] * rbuf[k];
					}
				}
			}
		}
	}
	/*Mono and Stereo*/
	else{
		for(int i = 0; i < 2; i++){
			for(size_t j = 0; j < oframes; j++){
				obuf[j * 2 + i] = 0;
				for(size_t k = 0; k < rframes; k++){
					if( (j - k) >= 0 && (j - k) <= iframes){
						obuf[j * 2 + i] += ibuf[j - k] * rbuf[k * 2 + i];
					}
				}
			}
		}
	}
	
}
void maxScale(float *ibuf, long long iframes, long long oSize, float *obuf){
	/*Find peak of input*/
	float peak_in = 0;
    for (long long i = 0; i < iframes; i++) {
        if (std::abs(ibuf[i]) > peak_in){
			peak_in = std::abs(ibuf[i]);
		}
    }

	/*Find peak of output*/
	float peak_out = 0;
    for (long long i = 0; i < oSize; i++) {
        if (std::abs(obuf[i]) > peak_out){
			peak_out = std::abs(obuf[i]);
		}
    }
	
	/*Scale accordingly*/
	float scale = peak_in/peak_out;
	for (long long i = 0; i < oSize; i++) {
        obuf[i] *= scale;
    }   
}