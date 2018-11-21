#include "Main.cuh"

float *gpuEntry(std::string input, std::string reverb, std::string out, bool timeDomain) {
	setlocale(LC_NUMERIC, "");
	bool blockProcessingOn = false;
	/*Forward variable declarations*/
	float *obuf;
	float *buf, *rbuf;
	long long i_size = 0, r_size = 0, o_size = 0, new_size = 0;
	int iCh = 0, iSR = 0, rCh = 0, rSR = 0;
	/*Obtain audio block size based off the GPU specs*/
	long long audioBlockSize = getAudioBlockSize();
	readFileExperimental3(input.c_str(), reverb.c_str(),
		&iCh, &iSR, &i_size, &rCh, &rSR, &r_size, 
		&buf, &rbuf, &new_size, &blockProcessingOn, timeDomain);
	
	o_size = i_size + r_size - 1;
	
	if(timeDomain){
		//float * TDconvolution(float ** buf, float ** rbuf, long long new_size, long long old_size, long long written_size);
		//obuf = TDconvolution(&buf, &rbuf, new_size, i_size, o_size);
		obuf = timeDomainConvolutionExperimental(&buf, &rbuf, i_size, o_size);
	}
	else{
		if(blockProcessingOn){
			int numDevs = 1;
			cudaGetDeviceCount(&numDevs);
			if(numDevs == 1){
				obuf = blockConvolution(&buf, &rbuf, i_size, o_size, audioBlockSize);
			}
			else{
				obuf = multiGPUFFT(buf, rbuf, i_size, r_size);
			}
		}
		else
			obuf = convolution(&buf, &rbuf, new_size, i_size, o_size);
	}
	if (out.c_str()[0] != ' '){
		if (obuf != NULL){
			//fprintf(stderr, "Writing output file %s\n", out.c_str());
			writeFile(out.c_str(), obuf, o_size, iSR, iCh);
		}
	}
	
	return obuf;
}

