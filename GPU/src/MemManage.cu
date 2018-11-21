#include "MemManage.cuh"

__global__ void FillWithZeros(float *buf, long long start, long long size) {
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID + start < size)
		buf[threadID + start] = 0.0f;
}

void printSize() {
	size_t free = 0, total = 0;
	checkCudaErrors(cudaMemGetInfo(&free, &total));
	fprintf(stderr, "GPU Global Memory Stats: Size Free: %.2fMB\tSize Total: %.2fMB\tSize Used: %.2fMB\n", free / 1048576.0f, total / 1048576.0f, (total - free) / 1048576.0f);
}
size_t getTotalSize() {
	size_t free = 0, total = 0;
	checkCudaErrors(cudaMemGetInfo(&free, &total));
	return total;
}
size_t getFreeSize() {
	size_t free = 0, total = 0;
	checkCudaErrors(cudaMemGetInfo(&free, &total));
	return free;
}