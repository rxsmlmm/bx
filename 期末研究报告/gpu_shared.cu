#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "common.h"

// 声明外部函数（定义在 gpu_basic.cu 中）
extern __global__ void division_kernel(float* data, int k, int n);
extern void partial_pivoting(std::vector<float>& aug, int k, int n);

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 核函数：消去下方行（共享内存优化版）
__global__ void eliminate_kernel_shared(float* data, int k, int n) {
    __shared__ float pivot_row[4096];
    
    int row = k + 1 + blockIdx.x;
    if (row >= n) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int cols = n + 1;
    
    // 将主元行加载到共享内存（包含 b 列）
    for (int j = k + tid; j < cols; j += stride) {
        pivot_row[j] = data[k * cols + j];
    }
    __syncthreads();
    
    float factor = data[row * cols + k];
    data[row * cols + k] = 0.0f;
    
    // 从共享内存读取主元行进行消去（包含 b 列）
    for (int col = k + 1 + tid; col < cols; col += stride) {
        data[row * cols + col] -= factor * pivot_row[col];
    }
}

void gpu_shared_gaussian(std::vector<float>& A, std::vector<float>& b, 
                         std::vector<float>& x, int n) {
    int cols = n + 1;
    int aug_size = n * cols;
    std::vector<float> aug_host(aug_size);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug_host[i * cols + j] = A[i * n + j];
        }
        aug_host[i * cols + n] = b[i];
    }
    
    for (int k = 0; k < n; k++) {
        partial_pivoting(aug_host, k, n);
    }
    
    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, aug_size * sizeof(float)));
    
    // ========== 创建传输计时事件 ==========
    hipEvent_t ev_h2d_start, ev_h2d_stop;
    hipEvent_t ev_kernel_start, ev_kernel_stop;
    hipEvent_t ev_d2h_start, ev_d2h_stop;
    
    HIP_CHECK(hipEventCreate(&ev_h2d_start));
    HIP_CHECK(hipEventCreate(&ev_h2d_stop));
    HIP_CHECK(hipEventCreate(&ev_kernel_start));
    HIP_CHECK(hipEventCreate(&ev_kernel_stop));
    HIP_CHECK(hipEventCreate(&ev_d2h_start));
    HIP_CHECK(hipEventCreate(&ev_d2h_stop));
    
    // ========== H2D 传输计时 ==========
    HIP_CHECK(hipEventRecord(ev_h2d_start, 0));
    HIP_CHECK(hipMemcpy(d_data, aug_host.data(), aug_size * sizeof(float), 
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(ev_h2d_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_h2d_stop));
    
    // ========== Kernel 计时 ==========
    int threads_per_block = 256;
    
    HIP_CHECK(hipEventRecord(ev_kernel_start, 0));
    
    for (int k = 0; k < n; k++) {
        int cols_remain = cols - k;
        int blocks_division = (cols_remain + threads_per_block - 1) / threads_per_block;
        int rows_eliminate = n - (k + 1);
        int blocks_eliminate = rows_eliminate;
        
        hipLaunchKernelGGL(division_kernel, dim3(blocks_division), dim3(threads_per_block), 0, 0, d_data, k, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        
        if (rows_eliminate > 0) {
            hipLaunchKernelGGL(eliminate_kernel_shared, dim3(blocks_eliminate), dim3(threads_per_block), 0, 0, d_data, k, n);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        }
    }
    
    HIP_CHECK(hipEventRecord(ev_kernel_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_kernel_stop));
    
    // ========== D2H 传输计时 ==========
    std::vector<float> aug_result(aug_size);
    
    HIP_CHECK(hipEventRecord(ev_d2h_start, 0));
    HIP_CHECK(hipMemcpy(aug_result.data(), d_data, aug_size * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(ev_d2h_stop, 0));
    HIP_CHECK(hipEventSynchronize(ev_d2h_stop));
    
    // ========== 读取时间 ==========
    float h2d_ms, kernel_ms, d2h_ms;
    HIP_CHECK(hipEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_stop));
    HIP_CHECK(hipEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));
    
    std::cout << "[GPU Shared] H2D: " << h2d_ms << " ms, Kernel: " << kernel_ms << " ms, D2H: " << d2h_ms << " ms" << std::endl;
    std::cout << "[GPU Shared] Total: " << (h2d_ms + kernel_ms + d2h_ms) << " ms" << std::endl;
    
    // ========== 销毁事件 ==========
    HIP_CHECK(hipEventDestroy(ev_h2d_start));
    HIP_CHECK(hipEventDestroy(ev_h2d_stop));
    HIP_CHECK(hipEventDestroy(ev_kernel_start));
    HIP_CHECK(hipEventDestroy(ev_kernel_stop));
    HIP_CHECK(hipEventDestroy(ev_d2h_start));
    HIP_CHECK(hipEventDestroy(ev_d2h_stop));
    
    // ========== 回代 ==========
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
}