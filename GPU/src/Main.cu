#include "Main.cuh"
void drawGraph(float * buf, long long size, int SR, const char * name){
	Print("Taking HELLA LONG to plot the output\n");
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
void printMe(passable *p){
	fprintf(stderr, "\nAddress of p: %p\n", (void*)p);
	fprintf(stderr, "Address of p->input: %p\n", p->input);
	fprintf(stderr, "Address of p->reverb: %p\n", p->reverb);
	fprintf(stderr, "Address of p->input->d_buf: %p\n", p->input->d_buf);
	fprintf(stderr, "Address of p->reverb->d_buf: %p\n", p->reverb->d_buf);
	fprintf(stderr, "Address of p->input->buf: %p\n", p->input->buf);
	fprintf(stderr, "Address of p->reverb->buf: %p\n", p->reverb->buf);
	fprintf(stderr, "iFrames: %'lli\n", p->input->frames);
	fprintf(stderr, "rFrames: %'lli\n", p->reverb->frames);
	fprintf(stderr, "paddedSize: %'lli\n\n", p->paddedSize);
}
__global__ void doNothing(){}

float *gpuEntry(std::string input, std::string reverb, std::string out, bool timeDomain) {
	setlocale(LC_NUMERIC, "");
	/*Forward variable declarations*/
	passable *p;
	float *obuf;
	long long oFrames;
	int oCh = 1;
	int SR = 0;
	bool blockProcessingOn = false;
	doNothing<<<1, 1>>>();
	p = (passable*)malloc(sizeof(struct passable));
	p->input = (audio_container*)malloc(sizeof(struct audio_container));
	p->reverb = (audio_container*)malloc(sizeof(struct audio_container));
	
	readFileExperimentalDebug(input.c_str(), reverb.c_str(), &SR, &blockProcessingOn, timeDomain, p);
	oFrames = p->input->frames + p->reverb->frames - 1;
	if(p->type != mono_mono){
		oCh = 2;
	}
	printMe(p);
	if(timeDomain){
		obuf = TDconvolution(p);
	}
	else{
		if(blockProcessingOn){
			/*TODO: Stereo Support*/
			int numDevs = 1;
			cudaGetDeviceCount(&numDevs);
			if(numDevs == 1){
				obuf = blockConvolution(p);
			}
			else{
				obuf = multiGPUFFTDebug(p);
			}
		}
		else{
			Print("Running Convolution\n");
			obuf = convolution(p);
		}
	}
	if (obuf != NULL){
		if(p->type != mono_mono){
			// drawGraph(obuf, oFrames, SR, "out1.png");
			// drawGraph(obuf + oFrames, oFrames, SR, "out2.png");
			// writeFile("out1.wav", obuf, oFrames, SR, 1);
			// writeFile("out2.wav", obuf + p->paddedSize, oFrames, SR, 1);
			Print("Interleaving the file\n");

			/*Interleaving function in place doesn't work right now
			interleave(obuf, p->paddedSize);
			*/
			long long end = p->paddedSize;
			if(blockProcessingOn)
				end = oFrames;
			float *scrap = (float *)malloc(end * 2 * sizeof(float));
			for(long long i = 0; i < end* 2; i++){
				scrap[i] = obuf[i];
			}
			for(long long i = 0; i < end; i++){
				obuf[i * 2] = scrap[i];
				obuf[i * 2 + 1] = scrap[i + end];
			}
			free(scrap);
		}
		//drawGraph(obuf, oFrames * oCh, SR, "out3.png");
		if (out.c_str()[0] != ' '){
			
				fprintf(stderr, "Writing output file %s\n", out.c_str());
				writeFile(out.c_str(), obuf, oFrames, SR, oCh);
		}
	}
	return obuf;
}

