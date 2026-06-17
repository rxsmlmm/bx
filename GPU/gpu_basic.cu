#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "common.h"

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 核函数1：归一化主元行
__global__ void division_kernel(float* data, int k, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int cols = n + 1;
    float pivot = data[k * cols + k];
    
    if (fabsf(pivot) < 1e-12f) return;
    
    for (int col = k + tid; col < cols; col += total_threads) {
        data[k * cols + col] /= pivot;
    }
}

// 核函数2：消去下方行
__global__ void eliminate_kernel(float* data, int k, int n) {
    int row = k + 1 + blockIdx.x;
    if (row >= n) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int cols = n + 1;
    
    float factor = data[row * cols + k];
    data[row * cols + k] = 0.0f;
    
    for (int col = k + 1 + tid; col < cols; col += stride) {
        data[row * cols + col] -= factor * data[k * cols + col];
    }
}

// CPU 端部分主元选取
void partial_pivoting(std::vector<float>& aug, int k, int n) {
    int cols = n + 1;
    int max_row = k;
    float max_val = fabsf(aug[k * cols + k]);
    
    for (int i = k + 1; i < n; i++) {
        float val = fabsf(aug[i * cols + k]);
        if (val > max_val) {
            max_val = val;
            max_row = i;
        }
    }
    
    if (max_row != k) {
        for (int j = 0; j < cols; j++) {
            std::swap(aug[k * cols + j], aug[max_row * cols + j]);
        }
    }
}

void gpu_basic_gaussian(std::vector<float>& A, std::vector<float>& b, 
                        std::vector<float>& x, int n) {
    int cols = n + 1;
    int aug_size = n * cols;
    std::vector<float> aug_host(aug_size);
    
    // 构建增广矩阵
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug_host[i * cols + j] = A[i * n + j];
        }
        aug_host[i * cols + n] = b[i];
    }
    
    // 部分主元选取
    for (int k = 0; k < n; k++) {
        partial_pivoting(aug_host, k, n);
    }
    
    // 分配 GPU 内存并拷贝
    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, aug_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_data, aug_host.data(), aug_size * sizeof(float), 
                        hipMemcpyHostToDevice));
    
    int threads_per_block = 256;
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, 0));
    
    for (int k = 0; k < n; k++) {
        int cols_remain = cols - k;
        int blocks_division = (cols_remain + threads_per_block - 1) / threads_per_block;
        int rows_eliminate = n - (k + 1);
        int blocks_eliminate = rows_eliminate;
        
        hipLaunchKernelGGL(division_kernel, dim3(blocks_division), dim3(threads_per_block), 0, 0, d_data, k, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());    

        if (rows_eliminate > 0) {
            hipLaunchKernelGGL(eliminate_kernel, dim3(blocks_eliminate), dim3(threads_per_block), 0, 0, d_data, k, n);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float kernel_time_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&kernel_time_ms, start, stop));
    std::cout << "[GPU Basic] Kernel time: " << kernel_time_ms << " ms" << std::endl;
    
    std::vector<float> aug_result(aug_size);
    HIP_CHECK(hipMemcpy(aug_result.data(), d_data, aug_size * sizeof(float),
                        hipMemcpyDeviceToHost));
    
    x.resize(n);
    x[n - 1] = aug_result[(n - 1) * cols + n];  
    for (int i = n - 2; i >= 0; i--) {
        float sum = aug_result[i * cols + n];
        for (int j = i + 1; j < n; j++) {
            sum -= aug_result[i * cols + j] * x[j];
        }
        x[i] = sum;  
    }
    
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}