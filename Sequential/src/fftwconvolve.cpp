#include "fftwconvolve.h"
enum flags {
    mono_mono,
    mono_stereo,
    stereo_mono,
    stereo_stereo,
};
void pointwiseMultiplication(fftwf_complex *f_x, fftwf_complex *f_h,  long long paddedFrames, enum flags num){
    /*Do pointwise multiplication for convolution*/
    int iCh = num <= 1 ? 1 : 2;
    int rCh = num == mono_stereo || num == stereo_stereo ? 2 : 1;
    if (num == mono_mono){
        for(long long i = 0; i < iCh * (paddedFrames / 2 + 1); i++){
            fftwf_complex temp;
            temp[0] = f_x[i][0];
            temp[1] = f_x[i][1];
            f_x[i][0] = temp[0] * f_h[i][0] - temp[1] * f_h[i][1];
            f_x[i][1] = temp[0] * f_h[i][1] + temp[1] * f_h[i][0];
        }
    }
    else if(num <= mono_stereo){
        for(long long i = 0; i < paddedFrames / 2 + 1; i++){
          fftwf_complex temp;
            temp[0] = f_h[i][0];
            temp[1] = f_h[i][1];
            f_h[i][0] = f_x[i][0] * temp[0] - f_x[i][1] * temp[1];
            f_h[i][1] = f_x[i][0] * temp[1] + f_x[i][1] * temp[0];

            temp[0] = f_h[i + (paddedFrames/2 +1)][0];
            temp[1] = f_h[i + (paddedFrames/2 +1)][1];
            f_h[i + (paddedFrames/2 + 1)][0] = f_x[i][0] * temp[0] - f_x[i][1] * temp[1];
            f_h[i + (paddedFrames/2 + 1)][1] = f_x[i][0] * temp[1] + f_x[i][1] * temp[0];
        }
    }
    else{
        for(long long i = 0; i < paddedFrames / 2 + 1; i++){
            fftwf_complex temp;
            temp[0] = f_x[i][0];
            temp[1] = f_x[i][1];
            f_x[i][0] = temp[0] * f_h[i][0] - temp[1] * f_h[i][1];
            f_x[i][1] = temp[0] * f_h[i][1] + temp[1] * f_h[i][0];

            temp[0] = f_x[i + (paddedFrames/2 +1)][0];
            temp[1] = f_x[i + (paddedFrames/2 +1)][1];
            f_x[i + (paddedFrames/2 +1)][0] = temp[0] * f_h[i][0] - temp[1] * f_h[i][1];
            f_x[i + (paddedFrames/2 +1)][1] = temp[0] * f_h[i][1] + temp[1] * f_h[i][0];
        }
    }
}
void scaleArray(float *x, float peak_in, long long size){
    /*Find peak of output*/
	float peak_out = 0;
    for (long long i = 0; i < size; i++) {
        if (std::abs(x[i] ) > peak_out){
			peak_out = std::abs(x[i]);
		}
    }
	/*Scale accordingly*/
	float scale = peak_in/peak_out;
	for (long long i = 0; i < size ; i++) {
        x[i] *= scale;
    }
}
/*float *x  & *h are fftwf_real, pre-padded. Returns whichever adress contains the data*/
float * simpleConvolve(float *x, float *h, int paddedFrames, enum flags num){
    int outPaddedSamples = num == mono_mono ? paddedFrames : (int)(paddedFrames * 2L);
    fftwf_complex *f_x, *f_h;
    
    int iCh = num <= 1 ? 1 : 2;
    int rCh = num == mono_stereo || num == stereo_stereo ? 2 : 1;
    
    /*Allocate complex arrays*/
    f_x = (fftwf_complex*) fftwf_alloc_complex(iCh * (paddedFrames / 2 + 1));
    f_h = (fftwf_complex*) fftwf_alloc_complex(rCh * (paddedFrames / 2 + 1));
    
    fftwf_plan mono_input, interleaved_left, interleaved_right, left_out, right_out, mono_out;
    mono_input = fftwf_plan_dft_r2c_1d(outPaddedSamples, x, f_x, FFTW_ESTIMATE);
    mono_out = fftwf_plan_dft_c2r_1d(outPaddedSamples, f_x, x, FFTW_ESTIMATE);

    interleaved_left = fftwf_plan_many_dft_r2c(1, &paddedFrames, 1, x, NULL, 2, 0, f_x, NULL, 1, 0, FFTW_ESTIMATE);
    interleaved_right = fftwf_plan_many_dft_r2c(1, &paddedFrames, 1, x + 1, NULL, 2, 0, f_x + (paddedFrames / 2 + 1), NULL, 1, 0, FFTW_ESTIMATE);
    left_out = fftwf_plan_many_dft_c2r(1, &paddedFrames, 1, f_x, NULL, 1, 0, x, NULL, 2, 0, FFTW_ESTIMATE);
    right_out = fftwf_plan_many_dft_c2r(1, &paddedFrames, 1, f_x + (paddedFrames / 2 + 1), NULL, 1, 0, x + 1, NULL, 2, 0, FFTW_ESTIMATE);
    

    /*Transform input and reverb*/
    switch(num){
        case mono_mono:
            fftwf_execute_dft_r2c(mono_input, x, f_x);
            fftwf_execute_dft_r2c(mono_input, h, f_h); 
            break;
        case mono_stereo:
            fftwf_execute_dft_r2c(mono_input, x, f_x);
            fftwf_execute_dft_r2c(interleaved_left, h, f_h);
            fftwf_execute_dft_r2c(interleaved_right, h + 1, f_h + (paddedFrames / 2 + 1));
            break;
        case stereo_mono:
            fftwf_execute(interleaved_left);
            fftwf_execute(interleaved_right);
            fftwf_execute_dft_r2c(mono_input, h, f_h); 
            break;
        case stereo_stereo:
            fftwf_execute(interleaved_left);
            fftwf_execute(interleaved_right);
            fftwf_execute_dft_r2c(interleaved_left, h, f_h);
            fftwf_execute_dft_r2c(interleaved_right, h + 1, f_h + (paddedFrames / 2 + 1));
            break;
    }
    pointwiseMultiplication(f_x, f_h, paddedFrames, num);
    
    /*Inverse transformation of the output*/
    if(num == mono_mono){
        fftwf_execute(mono_out);
    }
    else if (num >= stereo_mono){
        fftwf_execute(left_out);
        fftwf_execute(right_out);
    }
    else{
        fftwf_execute_dft_c2r(left_out, f_h, x);
        fftwf_execute_dft_c2r(right_out, f_h + (paddedFrames / 2 + 1), x + 1);      
    }
    fftwf_destroy_plan(interleaved_left);
    fftwf_destroy_plan(interleaved_right);
    fftwf_destroy_plan(mono_input);
    fftwf_destroy_plan(left_out);
    fftwf_destroy_plan(right_out);
    fftwf_destroy_plan(mono_out);
    fftwf_free(f_x); 
    fftwf_free(f_h);
    if (num == mono_stereo){
        return h;
    }
    return x;
}

float * convolve(float *ibuf, float *rbuf, long long iFrames, long long rFrames, long long paddedFrames, enum flags num){
    long long outPaddedSamples = num == mono_mono ? paddedFrames : paddedFrames * 2L;
    int iCh = num <= 1 ? 1 : 2;
    int rCh = num == mono_stereo || num == stereo_stereo ? 2 : 1;

    /*Allocate padded input*/
    float *x, *h;
    x = fftwf_alloc_real(outPaddedSamples);
    if(outPaddedSamples > __INT_MAX__ || x == NULL){
        if (x != NULL) fftwf_free(x);
        int mod = iFrames % 2;
        int newiFrames = iFrames / 2;
        int outFrames = rFrames + newiFrames - 1;
        int recursiveOutPaddedFrames = pow(2, ceil(log2(outFrames)));
        if (num != mono_mono) recursiveOutPaddedFrames *= 2;
        setlocale(LC_NUMERIC, "");
        
        float *returnBuf = (float*)malloc(outPaddedSamples * sizeof(float));
        if(returnBuf == NULL){
            fprintf(stderr, "ERROR: Cannot allocate memory for returnBuf\n");
            exit(EXIT_FAILURE);
        }
        float *obuf1 = convolve(ibuf, rbuf, newiFrames, rFrames, recursiveOutPaddedFrames, num);
        memcpy(returnBuf, obuf1, outFrames * sizeof(float));
        free(obuf1);
       float *obuf2 = convolve(&ibuf[newiFrames], rbuf, newiFrames + mod, rFrames, recursiveOutPaddedFrames, num);

        for(int i = 0; i < rFrames - 1; i++){
            returnBuf[newiFrames + i] += obuf2[i];
        }
        memcpy(returnBuf + outFrames, obuf2 + rFrames - 1, (outFrames - rFrames - 1 + mod) * sizeof(float));
        free(obuf2);
        return returnBuf;
    }
    
    /*Copy data into x*/
    memcpy(x, ibuf, iFrames * iCh * sizeof(float));
    
    /*Allocate padded reverb*/
    h = fftwf_alloc_real(outPaddedSamples);
    if(h == NULL){
        fprintf(stderr, "ERROR: Unable to allocate memory for h\n");
        exit(1);
    }
    /*Copy data into h*/
    memcpy(h, rbuf, rFrames * rCh * sizeof(float));

    /*Pad remaining values to 0*/
    for(long long i = rFrames * rCh; i < outPaddedSamples; i++){
        h[i] = 0.0f;
        if(i >= iFrames * iCh){
            x[i] = 0.0f;
        }
    }

    float *obuf = simpleConvolve(x, h, paddedFrames, num);

    /*Allocate memory for output*/
    float *out = (float*) malloc(outPaddedSamples * sizeof(float));

    memcpy(out, obuf, outPaddedSamples * sizeof(float));

    /*Destroy and free memory*/
    fftwf_free(x);
    fftwf_free(h);
    return out;
}
float *entry(float *ibuf, float *rbuf, long long iFrames, long long rFrames, int iCh, int rCh){
    float peak_in, peak_out, scale;
    int oCh = iCh == 2 || rCh == 2 ? 2 : 1;
    long long outFrames = rFrames + iFrames - 1;
    long long outSamples = oCh == 1 ? outFrames : outFrames * 2L;
    long long paddedFrames = pow(2, ceil(log2(outFrames)));
    long long outPaddedSamples = oCh == 1 ? paddedFrames : paddedFrames * 2L;
	peak_in = 0;
    for (long long i = 0; i < iFrames * iCh; i++) {
       float currNum = std::abs(ibuf[i]);
        if (currNum > peak_in){
			peak_in = currNum;
		}
    }

    flags num;
    if(oCh == 1){
        num == mono_mono;
    }
    else if(iCh == rCh){
        num = stereo_stereo;
    }
    else if(iCh == 1){
        num = mono_stereo;
    }
    else{
        num = stereo_mono;
    }
    
    float *out = convolve(ibuf, rbuf, iFrames, rFrames, paddedFrames, num);
    scaleArray(out, peak_in, outPaddedSamples);
    
    /*Free original ibuf*/
    free(ibuf);
	/*Free original rbuf*/
    free(rbuf);
    return out;
}