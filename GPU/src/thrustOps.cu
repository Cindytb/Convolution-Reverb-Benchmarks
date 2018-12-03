#include "thrustOps.cuh"

/*Functions to find extrema*/
float DExtrema(float *pointer, long long size){
	/*Convert raw float pointer into a thrust device pointer*/
	thrust::device_ptr<float> thrust_d_signal(pointer);
	
	thrust::pair < thrust::device_ptr<float>, thrust::device_ptr<float> >pair = 
		thrust::minmax_element(thrust::device, thrust_d_signal, thrust_d_signal + size);
	float *d_min, *d_max;
	float min = 0, max = 0;
	
	d_min = pair.first.get();
	d_max = pair.second.get();
	
	checkCudaErrors(cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDefault));
	checkCudaErrors(cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDefault));

	//float result = std::abs(*min) > *max ? std::abs(*min) : *max;
	//free(min);
	//free(max);
	//return result;
	return std::abs(min) > max ? std::abs(min) : max;
}
void fillWithZeroes(float **target_buf, long long old_size, long long new_size){
	thrust::device_ptr<float> dev_ptr(*target_buf);
	thrust::fill(dev_ptr + old_size, dev_ptr + new_size, (float) 0.0f);
}