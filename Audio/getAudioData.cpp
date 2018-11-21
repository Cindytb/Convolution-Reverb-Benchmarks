#include <stdio.h>
#include <string.h>
#include <sndfile.h>


int main(int argc, char **argv){
    if(argc != 2){
        printf("Usage: %s filename.wav\n", argv[0]);
        return 1;
    }
    SF_INFO info;
	SNDFILE *sndfile;
	memset(&info, 0, sizeof(info));

	/*Read input*/
	sndfile = sf_open(argv[1], SFM_READ, &info);
	if (sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", argv[1]);
		return 1;
	}
	printf("Opened file '%s'\n", argv[1]);
	printf("Sample rate : %d\n", info.samplerate);
	printf("Channels    : %d\n", info.channels);
    printf("Frames      : %ld\n", info.frames);


    // SndfileHandle file ;
	// file = SndfileHandle (argv[1]);
    // printf("Opened file '%s'\n", argv[1]);
	// printf("Sample rate : %d\n", file.samplerate());
	// printf("Channels    : %d\n", file.channels());
    // printf("Frames      : %li\n", file.frames());
    sf_close(sndfile);
    return 0;
}