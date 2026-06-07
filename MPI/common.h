#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>

inline void generate_test_data(std::vector<float>& A, 
                                std::vector<float>& b, 
                                int n) {
    A.resize(n * n);
    b.resize(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i * n + j] = dis(gen);
                row_sum += std::fabs(A[i * n + j]);
            }
        }
        
        A[i * n + i] = row_sum + std::fabs(dis(gen)) + 1.0f;
        b[i] = dis(gen);
    }
}

inline float compute_residual(const std::vector<float>& A,
                              const std::vector<float>& x,
                              const std::vector<float>& b,
                              int n) {
    float residual = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = 0.0f;
        for (int j = 0; j < n; j++) {
            ax += A[i * n + j] * x[j];
        }
        residual += std::fabs(ax - b[i]);
    }
    return residual;
}

#endif