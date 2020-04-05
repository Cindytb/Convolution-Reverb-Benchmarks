
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../GPU/src/thrustOps.cuh"
#include <stdio.h>
#include <random>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
int thrustOpsTest(char* filename);
int systemTest(char * filename);
int sgDevConvolutionTest(char* filename);
int main()
{
	int retval = 0;

	printf("Testing thrustOps.cu\n");
	retval += thrustOpsTest("thrustOps.cu");

	printf("Testing GPU.exe\n");
	retval += systemTest("GPU.exe");


	return retval;
}
std::vector<float> generateSineSweep(float T, float f0, float f1, int fs) {
	int num_samples = T * fs;
	std::vector<float> time_axis(num_samples);
	for (int i = 0; i < num_samples; i++) {
		time_axis[i] = float(i) / fs;
	}
	float start_total = f0 / T;
	float sweep_rate = f1 / f0;
	float L = log(sweep_rate);
	std::vector<float> sweep(num_samples);
	for (int i = 0; i < num_samples; i++) {
		float freq = start_total / L * pow(sweep_rate, time_axis[i] / T) - 1;
		sweep[i] = sin(2 * M_PI * freq);
	}
	return sweep;

}
int conditional(bool status, char* filename, char* function_name) {
	int retval = 0;
	if (status) {
		printf("SUCCESS: ");
	}
	else {
		printf("FAILURE: ");
		retval |= 1;
	}
	printf("%s - %s\n", filename, function_name);
	return retval;
}
int thrustOpsTest(char *filename) {
	int size = 1048576;
	float* a = new float[size];
	float* d_a;
	int retval = 0;

	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, size);


	for (int i = 0; i < size; i++) {
		a[i] = dist(e2);
	}
	bool status = true;
	for (int i = 1; i < size; i++) {
		if (a[i] != a[0]) {
			status = false;
			break;
		}
	}
	if (status) {
		printf("TESTER ERROR: All numbers in the array are equal to %f\n", a[0]);
	}
	checkCudaErrors(cudaMalloc(&d_a, size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice));

	fillWithZeroes(&d_a, 0, size);

	checkCudaErrors(cudaMemcpy(a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost));

	status = true;
	for (int i = 0; i < size; i++) {
		if (fabs(a[i]) > std::numeric_limits<float>::epsilon()) {
			status = false;
			break;
		}
	}
	retval += conditional(status, filename, "fillWithZeros()");

	float randn = dist(e2);
	a[int(dist(e2))] = randn;
	checkCudaErrors(cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
	float max = DExtrema(d_a, size);
	retval += conditional(max == randn, filename, "DExtrema()");


	checkCudaErrors(cudaFree(d_a));
	delete[] a;
	return retval;
}

int sgDevConvolutionTest(char* filename) {
	return 0;
}
int spawnChildProcess(std::string cmd) {
	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Start the child process. 
	if (!CreateProcess(NULL,   // No module name (use command line)
		const_cast<char*>(cmd.c_str()),        // Command line
		NULL,           // Process handle not inheritable
		NULL,           // Thread handle not inheritable
		FALSE,          // Set handle inheritance to FALSE
		0,              // No creation flags
		NULL,           // Use parent's environment block
		NULL,           // Use parent's starting directory 
		&si,            // Pointer to STARTUPINFO structure
		&pi)           // Pointer to PROCESS_INFORMATION structure
		)
	{
		printf("CreateProcess failed (%d).\n", GetLastError());
		return 1;
	}

	// Wait until child process exits.
	WaitForSingleObject(pi.hProcess, INFINITE);
	// Close process and thread handles. 
	DWORD exit_code;
	GetExitCodeProcess(pi.hProcess, &exit_code);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	return int(exit_code);
}
int systemTest(char* filename) {

	int retval = 0;
	std::string path("D:\\Projects\\CUDA\\Convolution-Reverb-Benchmarks");
	std::string exe = path + "\\x64\\Debug\\GPU.exe";
	std::string sgDevBlockFlags(" -i D:\\AllAudioEVER\\1015M.wav -r \"D:\\School\\College\\Senior Year\\Capstone\\Code\\IR\\96000\\Mono\\koli_snow_site1_1way_mono.wav\"");
	std::string regularFlags = " -i " + path + "\\Audio\\Ex96.wav -r " + path + "\\Audio\\480000.wav";
	std::string mono_stereo_flags = " -i " + path + "\\Audio\\channel_testing\\441\\Ex_441_mono.wav -r " + path + "\\Audio\\channel_testing\\441\\H0e090a.wav";
	std::string stereo_stereo_flags = " -i " + path + "\\Audio\\channel_testing\\441\\Ex_441_stereo.wav -r " + path + "\\Audio\\channel_testing\\441\\H0e090a.wav";
	
	retval += conditional(!spawnChildProcess(exe + regularFlags), filename, "regular");
	retval += conditional(!spawnChildProcess(exe + sgDevBlockFlags), filename, "sgDevBlock");
	retval += conditional(!spawnChildProcess(exe + mono_stereo_flags), filename, "mono_stereo");
	retval += conditional(!spawnChildProcess(exe + stereo_stereo_flags), filename, "stereo_stereo");
	
	return retval;
}