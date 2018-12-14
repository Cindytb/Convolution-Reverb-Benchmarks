# Convolution-Reverb-Benchmarks

This research project aims to offload the work of Digital Signal Processing (DSP) onto a different chip, the GPU, and to compare the performance results to that of a CPU. I am testing one of the most computationally expensive DSP operations - massive convolution - on a CPU and a GPU. The goal is to benchmark them and see at what point it's better to use which chip.

This is a benchmarking test for convolution reverb with single core/sequential code and a parallelized implementation using CUDA and cuFFT. This is in fulfillment of my Music Technology Undergraduate Capstone Project.