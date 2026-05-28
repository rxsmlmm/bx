#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "gauss_common.h"
#include "gauss_serial.h"
#include "gauss_pthread_bar.h"
#include "gauss_pthread_dyn.h"
#include "gauss_pthread_bar_simd.h"
#include "gauss_omp_simd.h"

// 测试一次，返回耗时(ms)
double test_once(void (*func)(std::vector<std::vector<float>>&, std::vector<float>&, std::vector<float>&, int),
                 int n, int threads) {
    std::vector<std::vector<float>> A;
    std::vector<float> b, x;
    generate_test_data(A, b, n);
    auto A_orig = A;
    auto b_orig = b;
    
    auto start = std::chrono::high_resolution_clock::now();
    func(A, b, x, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> elapsed = end - start;
    
    float residual = compute_residual(A_orig, x, b_orig);
    if (residual >= 1e-2f) return -1.0;  // 失败标记
    return elapsed.count();
}

// 串行版本 wrapper（忽略 threads 参数）
void serial_wrapper(std::vector<std::vector<float>>& A, std::vector<float>& b, std::vector<float>& x, int) {
    gaussian_elimination(A, b, x);
}

int main() {
    int sizes[] = {500, 1000, 1500};
    int threads_list[] = {1, 2, 4, 8};
    int repeat = 3;
    
    struct {
        const char* name;
        void (*func)(std::vector<std::vector<float>>&, std::vector<float>&, std::vector<float>&, int);
        bool is_parallel;
    } versions[] = {
        {"V1_Serial",           serial_wrapper,                     false},
        {"V3_Pthread_Bar",      gaussian_elimination_pthread_bar,   true},
        {"V2_Pthread_Dyn",      gaussian_elimination_pthread_dyn,   true},
        {"V8_Pthread_Bar_SIMD", gaussian_elimination_pthread_bar_simd, true},
        {"V9_OpenMP_SIMD",      gaussian_elimination_omp_simd,      true},
    };
    
    std::cout << "Platform,Version,Size,Threads,Time_ms" << std::endl;
    
    for (auto& ver : versions) {
        for (int n : sizes) {
            if (ver.is_parallel) {
                for (int t : threads_list) {
                    double sum = 0;
                    bool pass = true;
                    for (int r = 0; r < repeat; r++) {
                        double tms = test_once(ver.func, n, t);
                        if (tms < 0) { pass = false; break; }
                        sum += tms;
                    }
                    if (pass)
                        std::cout << "ARM," << ver.name << "," << n << "," << t << ","
                                  << std::fixed << std::setprecision(4) << sum / repeat << std::endl;
                }
            } else {
                double sum = 0;
                bool pass = true;
                for (int r = 0; r < repeat; r++) {
                    double tms = test_once(ver.func, n, 1);
                    if (tms < 0) { pass = false; break; }
                    sum += tms;
                }
                if (pass)
                    std::cout << "ARM," << ver.name << "," << n << ",,"
                              << std::fixed << std::setprecision(4) << sum / repeat << std::endl;
            }
        }
    }
    
    return 0;
}
