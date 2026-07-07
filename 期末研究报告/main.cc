#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "common.h"
#include "serial.h"

void gpu_basic_gaussian(std::vector<float>& A, std::vector<float>& b, 
                        std::vector<float>& x, int n);
void gpu_shared_gaussian(std::vector<float>& A, std::vector<float>& b, 
                         std::vector<float>& x, int n);

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    
    std::cout << "========== Gaussian Elimination GPU Test ==========" << std::endl;
    
    for (int n : sizes) {
        std::cout << "\\n========== n=" << n << " ==========" << std::endl;
        std::cout << std::left << std::setw(18) << "Version" 
                  << std::setw(14) << "Time(ms)" 
                  << std::setw(14) << "Residual" 
                  << "Correct" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        std::vector<float> A, b;
        generate_test_data(A, b, n);
        
        // ========== 串行 ==========
        std::vector<float> A_serial = A;
        std::vector<float> b_serial = b;
        std::vector<float> x_serial;
        
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_elimination(A_serial, b_serial, x_serial, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_serial = end - start;
        
        float residual_serial = compute_residual(A, x_serial, b, n);
        bool correct_serial = (residual_serial < 1e-1f);
        
        std::cout << std::left << std::setw(18) << "Serial"
                  << std::setw(14) << std::fixed << std::setprecision(2) << elapsed_serial.count()
                  << std::setw(14) << residual_serial
                  << (correct_serial ? "OK" : "FAIL") << std::endl;
        
        // ========== GPU 基础版本 ==========
        std::vector<float> A_gpu_basic = A;
        std::vector<float> b_gpu_basic = b;
        std::vector<float> x_gpu_basic;
        
        gpu_basic_gaussian(A_gpu_basic, b_gpu_basic, x_gpu_basic, n);
        
        float residual_gpu_basic = compute_residual(A, x_gpu_basic, b, n);
        bool correct_gpu_basic = (residual_gpu_basic < 1e-1f);
        
        std::cout << std::left << std::setw(18) << "GPU Basic"
                  << std::setw(14) << "(see above)"
                  << std::setw(14) << residual_gpu_basic
                  << (correct_gpu_basic ? "OK" : "FAIL") << std::endl;
        
        // ========== GPU 共享内存版本 ==========
        std::vector<float> A_gpu_shared = A;
        std::vector<float> b_gpu_shared = b;
        std::vector<float> x_gpu_shared;
        
        gpu_shared_gaussian(A_gpu_shared, b_gpu_shared, x_gpu_shared, n);
        
        float residual_gpu_shared = compute_residual(A, x_gpu_shared, b, n);
        bool correct_gpu_shared = (residual_gpu_shared < 1e-1f);
        
        std::cout << std::left << std::setw(18) << "GPU Shared"
                  << std::setw(14) << "(see above)"
                  << std::setw(14) << residual_gpu_shared
                  << (correct_gpu_shared ? "OK" : "FAIL") << std::endl;
    }
    
    return 0;
}