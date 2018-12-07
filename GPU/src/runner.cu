#include "Main.cuh"

int main(int argc, char **argv){
    std::string input = "../Audio/Ex96.wav";
	std::string reverb = "../Audio/480000.wav";
	
	std::string out(" ");
	
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
    float *obuf = gpuEntry(input, reverb, out, timeDomain);
	if (obuf == NULL){
		fprintf(stderr, "ERROR\n");
		return 100;
	}
    checkCudaErrors(cudaFreeHost(obuf));
    return 0;
}