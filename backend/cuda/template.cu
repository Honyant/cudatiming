#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

// CPU matrix multiplication reference implementation
void matrixMultCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Basic CUDA matrix multiplication kernel
__global__ void matrixMultGPU(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Verify results
bool verifyResults(const float* cpuResult, const float* gpuResult, int N, float tolerance = 1e-5) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(cpuResult[i] - gpuResult[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i << ": CPU = " 
                      << cpuResult[i] << ", GPU = " << gpuResult[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations>" << std::endl;
        return 1;
    }
    
    int N = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]);
    
    // Allocate and initialize host memory
    size_t matrixSize = N * N * sizeof(float);
    float *h_A = (float*)malloc(matrixSize);
    float *h_B = (float*)malloc(matrixSize);
    float *h_C_CPU = (float*)malloc(matrixSize);
    float *h_C_GPU = (float*)malloc(matrixSize);
    
    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // CPU matrix multiplication
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMultCPU(h_A, h_B, h_C_CPU, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    
    // Setup kernel execution parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    // Warm-up GPU
    matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // GPU matrix multiplication timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float totalGpuTime = 0.0f;
    
    for (int iter = 0; iter < iterations; ++iter) {
        cudaEventRecord(start);
        matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalGpuTime += milliseconds;
    }
    
    float avgGpuTime = totalGpuTime / iterations;
    
    // Copy result back to host
    cudaMemcpy(h_C_GPU, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = verifyResults(h_C_CPU, h_C_GPU, N);
    
    // Output results as JSON
    std::cout << "{" << std::endl;
    std::cout << "  \"matrix_size\": " << N << "," << std::endl;
    std::cout << "  \"iterations\": " << iterations << "," << std::endl;
    std::cout << "  \"cpu_time_ms\": " << cpu_duration.count() << "," << std::endl;
    std::cout << "  \"gpu_time_ms\": " << avgGpuTime << "," << std::endl;
    std::cout << "  \"speedup\": " << (cpu_duration.count() / avgGpuTime) << "," << std::endl;
    std::cout << "  \"verification\": " << (correct ? "true" : "false") << std::endl;
    std::cout << "}" << std::endl;
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    
    return 0;
}