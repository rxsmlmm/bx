#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include "gauss_common.h"
#include "gauss_omp.h"

double test_once(int n, int threads) {
    std::vector<std::vector<float>> A;
    std::vector<float> b, x;
    generate_test_data(A, b, n);
    auto A_orig = A;
    auto b_orig = b;
    
    auto start = std::chrono::high_resolution_clock::now();
    gaussian_elimination_omp(A, b, x, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> elapsed = end - start;
    
    float residual = compute_residual(A_orig, x, b_orig);
    if (residual >= 1e-2f) return -1.0;
    return elapsed.count();
}

int main(int argc, char* argv[]) {
    #ifdef VERSION_NAME
    const char* name = VERSION_NAME;
#else
    const char* name = (argc >= 2) ? argv[1] : "OpenMP";
#endif
    int sizes[] = {500, 1000, 1500};
    int threads_list[] = {1, 2, 4, 8};
    int repeat = 3;
    
    std::cout << "Platform,Version,Size,Threads,Time_ms" << std::endl;
    
    for (int n : sizes) {
        for (int t : threads_list) {
            double sum = 0;
            for (int r = 0; r < repeat; r++) {
                double tms = test_once(n, t);
                if (tms < 0) { sum = -1; break; }
                sum += tms;
            }
            if (sum >= 0)
                std::cout << "ARM," << name << "," << n << "," << t << ","
                          << std::fixed << std::setprecision(4) << sum / repeat << std::endl;
        }
    }
    return 0;
}