#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include "common.h"
#include "serial.h"

void gpu_basic_gaussian(std::vector<float>& A, std::vector<float>& b, 
                        std::vector<float>& x, int n);

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};  
    
    std::cout << "========== Gaussian Elimination GPU Test ==========" << std::endl;
    
    for (int n : sizes) {
        std::cout << "\nn=" << n << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::left << std::setw(15) << "Version" 
                  << std::setw(12) << "Time(ms)" 
                  << std::setw(12) << "Residual" 
                  << "Correct" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // 生成测试数据
        std::vector<float> A, b;
        generate_test_data(A, b, n);
        
        // 串行版本
        std::vector<float> A_serial = A;
        std::vector<float> b_serial = b;
        std::vector<float> x_serial;
        
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_elimination(A_serial, b_serial, x_serial, n);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_serial = end - start;
        
        float residual_serial = compute_residual(A, x_serial, b, n);
        bool correct_serial = (residual_serial < 1e-1f);
        
        std::cout << std::left << std::setw(15) << "Serial"
                  << std::setw(12) << std::fixed << std::setprecision(2) << elapsed_serial.count()
                  << std::setw(12) << residual_serial
                  << (correct_serial ? "OK" : "FAIL") << std::endl;
        
        // GPU 基础版本
        std::vector<float> A_gpu = A;
        std::vector<float> b_gpu = b;
        std::vector<float> x_gpu;
        
        gpu_basic_gaussian(A_gpu, b_gpu, x_gpu, n);
        
        float residual_gpu = compute_residual(A, x_gpu, b, n);
        bool correct_gpu = (residual_gpu < 1e-1f);
        
        std::cout << std::left << std::setw(15) << "GPU Basic"
                  << std::setw(12) << "(see above)"
                  << std::setw(12) << residual_gpu
                  << (correct_gpu ? "OK" : "FAIL") << std::endl;
        
        if (correct_serial && correct_gpu) {
            std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
                      << (elapsed_serial.count() / elapsed_serial.count()) << "x" << std::endl;
        }
    }
    
    return 0;
}