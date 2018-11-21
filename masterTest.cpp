#include <stdio.h>
#include <math.h>
#include <sndfile.hh>
#include "GPU/src/Main.cuh"
#include "Sequential/src/main.h"

int main(int argc, char **argv){
    std::string input = "../Audio/Ex96.wav";
	std::string reverb = "../Audio/480000.wav";
	std::string out(" ");
	setlocale(LC_NUMERIC, "");
	bool timeDomain = false;
	/*Parse Input*/
	for(int i = 1; i < argc; i++){
		if(argv[i][0] == '-'){
			if(argv[i][1] == 't' || argv[i][1] == 'T'){
				timeDomain = true;
			}
			else if(argv[i][1] == 'i'){
				input = argv[i+1];
			}
			else if(argv[i][1] == 'r'){
				reverb = argv[i+1];
			}
			else if(argv[i][1] == 'o'){
				out = argv[i + 1];
			}
			else if(argv[i][1] == 'h'){
				printf("Usage: %s [-t] [-i input.wav] [-r reverb.wav] [-o output.wav]\n", argv[0]);
				printf("Default input : ../Audio/Ex96.wav\n");
				printf("Default reverb: ../Audio/480000.wav\n");
				printf("No output file by default. \n-t : do time domain convolution\n");
			}
		}
	}
	fprintf(stderr, "Input: %s\n", input.c_str());
	fprintf(stderr, "Reverb: %s\n", reverb.c_str());
	fprintf(stderr, "Output: %s\n", out.c_str());
	fprintf(stderr, "\n\n");
    
    SndfileHandle f1, f2;
    f1 = SndfileHandle (input.c_str());
    fprintf(stderr, "Samples     : %'li\n", f1.frames() * f1.channels());
    long long f1Samples = f1.frames();

    f2 = SndfileHandle(reverb.c_str());
    fprintf(stderr,"Samples     : %'li\n", f2.frames() * f2.channels());
    long long f2Samples = f2.frames();

    float *buf1, *buf2;
    float epsilon = 1e-5f;
    fprintf(stderr,"Epsilon for this program is %e\n", epsilon);
    
    fprintf(stderr,"\n\nGPU VERSION:\n\n");
    buf1 = gpuEntry(input, reverb, out, timeDomain);
    fprintf(stderr,"\n\nCPU VERSION:\n\n");
    buf2 = seqEntry(input, reverb, out, timeDomain);
    
    if(buf1 == NULL || buf2 == NULL){
        fprintf(stderr, "Exiting program\n");
        if(buf1 != NULL) checkCudaErrors(cudaFreeHost(buf1));
        if(buf2 != NULL) free(buf2);
        return 0;
    }
    long long size = f1Samples + f2Samples - 1;
    float *buf3 = (float *)malloc(size * sizeof(float));
    int max = 0;
    int max2 = max;
    for(long long i = 0; i < size; i++){
        buf3[i] = fabs(buf1[i] - buf2[i]);
        if (buf3[max] < fabs(buf3[i])){
            max2 = max;
            max = i;
        }
    }
    fprintf(stderr,"\nThe maximum difference between the two files is at sample %'i\n",  max);
    fprintf(stderr,"buf1[max] = %11.8f\n", buf1[max]);
    fprintf(stderr,"buf2[max] = %11.8f\n", buf2[max]);
    fprintf(stderr,"Difference= %E\n", buf3[max]);

    fprintf(stderr,"\nThe 2nd maximum difference between the two files is at sample %'i\n",  max2);
    fprintf(stderr,"buf1[max] = %11.8f\n", buf1[max2]);
    fprintf(stderr,"buf2[max] = %11.8f\n", buf2[max2]);
    fprintf(stderr,"Difference= %E\n", buf3[max2]);

    
    bool bTestResult = buf3[max] < epsilon;
    FILE *fp;
    fp = fopen("results.log", "a");
    fprintf(fp, "Input file: %s\n", input.c_str());
	fprintf(fp,"Precision Test : %s\n", bTestResult ? "PASS" : "FAIL");
    printf("Input file: %s\n", input.c_str());
	printf("Precision Test : %s\n", bTestResult ? "PASS" : "FAIL");

    fclose(fp);
    checkCudaErrors(cudaFreeHost(buf1));
    free(buf2);
    free(buf3);
}