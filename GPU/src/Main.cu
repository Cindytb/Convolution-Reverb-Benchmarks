#include "Main.cuh"
#ifndef WIN64
void drawGraph(float * buf, long long size, int SR, const char * name){
	int NUM_COMMANDS = 2;
	const char * commandsForGnuplot[] = {"set terminal pngcairo size 1366,768 ",
		"set title \"Good Luck.\""};
    /*Opens an interface that one can use to send commands as if they were typing into the
 	*gnuplot command line.
 	*/
    FILE * gnuplotPipe = popen ("gnuplot", "w");
    for (int i=0; i < NUM_COMMANDS; i++){
		if(i == 1) fprintf(gnuplotPipe, "set output '%s'\n", name);
        fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); //Send commands to gnuplot one by one.
    }
    fprintf(gnuplotPipe, "plot '-' \n");
    for (long long i = 0; i < size; i++){
      fprintf(gnuplotPipe, "%lf %lf\n", (float)i/SR, buf[i]);
	  if(i % 1000000 == 0){
		  fprintf(stderr, "i: %'lli\n", i);
	  }
    }
	fprintf(gnuplotPipe, "e");
	pclose(gnuplotPipe);
}
#endif
void printMe(passable *p){
	fprintf(stderr, "\nAddress of p: %p\n", (void*)p);
	fprintf(stderr, "Address of p->input: %p\n", p->input);
	fprintf(stderr, "Address of p->reverb: %p\n", p->reverb);
	fprintf(stderr, "Address of p->input->d_buf: %p\n", p->input->d_buf);
	fprintf(stderr, "Address of p->reverb->d_buf: %p\n", p->reverb->d_buf);
	fprintf(stderr, "Address of p->input->buf: %p\n", p->input->buf);
	fprintf(stderr, "Address of p->reverb->buf: %p\n", p->reverb->buf);
	fprintf(stderr, "iFrames: %lu\n", p->input->frames);
	fprintf(stderr, "rFrames: %lu\n", p->reverb->frames);
	fprintf(stderr, "paddedSize: %lli\n\n", p->paddedSize);
}

void vss(char** reverb_files, char** input_files, char* output_name, float *lfe, int lfe_length, int lfe_channels) {
	setlocale(LC_NUMERIC, "");
	/*Forward variable declarations*/
	passable* p;
	float* obuf;
	long long oFrames;
	int oCh = 1;
	int SR = 0;
	bool blockProcessingOn = false;
	p = (passable*)malloc(sizeof(struct passable));
	p->input = (audio_container*)malloc(sizeof(struct audio_container));
	p->reverb = (audio_container*)malloc(sizeof(struct audio_container));

	int outputLength = 8388608 * 2;
	float* output = new float[outputLength];
	for (int i = 0; i < outputLength; i++) {
		output[i] = 0.0f;
	}
	for (int i = 0; i < 5; i++) {
		readFileExperimentalDebug(input_files[i], reverb_files[i], &SR, &blockProcessingOn, false, p);
		oFrames = p->input->frames + p->reverb->frames - 1;
		if (p->type != mono_mono) {
			oCh = 2;
		}

		Print("Running Convolution\n");
		obuf = specialConvolution(p);

		if (obuf != NULL) {
			if (p->type != mono_mono) {
				Print("Interleaving the file\n");

				/*Interleaving function in place doesn't work right now
				interleave(obuf, p->paddedSize);
				*/
				long long end = p->paddedSize;
				if (blockProcessingOn)
					end = oFrames;
				float* scrap = (float*)malloc(end * 2 * sizeof(float));
				for (long long i = 0; i < end * 2; i++) {
					scrap[i] = obuf[i];
				}
				for (long long i = 0; i < end; i++) {
					obuf[i * 2] = scrap[i];
					obuf[i * 2 + 1] = scrap[i + end];
				}
				free(scrap);
			}
		}
		/*Adding this convolved output to the final mix*/
		for (int j = 0; j < p->paddedSize * 2; j++) {
			output[j] += obuf[j];
		}

		checkCudaErrors(cudaFreeHost(obuf));
	}
	/*Adding the LFE channel*/
	for (int j = 0; j < lfe_length; j++) {
		if (lfe_channels == 2) {
			output[j] += lfe[j];
		}
		else {
			output[2 * j] += lfe[j];
			output[2 * j + 1] += lfe[j];
		}
	}
	/*Peak normalizing*/
	float maxVal = 0;
	for (int j = 0; j < outputLength; j++) {
		if (fabs(output[j]) > maxVal) {
			maxVal = fabs(output[j]);
		}
	}
	for (int j = 0; j < outputLength; j++) {
		output[j] /= maxVal;
	}
	writeFile(output_name, output, oFrames, SR, oCh);
	free(p->input);
	free(p->reverb);
	free(p);
	delete[] output;
}

float *gpuEntry(std::string input, std::string reverb, std::string out, bool timeDomain) {
	char *inputs[] = {"C_Stereo.wav", "Lf_Stereo.wav", "Rf_Stereo.wav", "Ls_Stereo.wav", "Rs_Stereo.wav"};
	char other_input[] = "LFE_Stereo.wav";
	
	/*Opening the LFE wave file*/
	SndfileHandle file;
	file = SndfileHandle(other_input);
	int lfe_length = file.frames() * file.channels();
	float *lfe = new float[lfe_length];
	file.read(lfe, lfe_length);

	char *reverbs[] = {"Cindy_C.wav", "Cindy_Lf.wav", "Cindy_Rf.wav", "Cindy_Ls.wav", "Cindy_Rs.wav"};
	char *reverbs2[] = { "Sarah_C.wav", "Sarah_Lf.wav", "Sarah_Rf.wav", "Sarah_Ls.wav", "Sarah_Rs.wav" };
	char* reverbs3[] = { "Cindy_BYPASS_C.wav", "Cindy_BYPASS_Lf.wav", "Cindy_BYPASS_Rf.wav", "Cindy_BYPASS_Ls.wav", "Cindy_BYPASS_Rs.wav" };
	char* reverbs4[] = { "Sarah_BYPASS_C.wav", "Sarah_BYPASS_Lf.wav", "Sarah_BYPASS_Rf.wav", "Sarah_BYPASS_Ls.wav", "Sarah_BYPASS_Rs.wav" };
	
	vss(reverbs, inputs, "Cindy.wav", lfe, lfe_length, file.channels());
	vss(reverbs2, inputs, "Sarah.wav", lfe, lfe_length, file.channels());
	vss(reverbs3, inputs, "Cindy_BYPASS.wav", lfe, lfe_length, file.channels());
	vss(reverbs4, inputs, "Sarah_BYPASS.wav", lfe, lfe_length, file.channels()); 
	
	
	return 0;
}

