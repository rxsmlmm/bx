#ifndef GAUSS_LIB_H
#define GAUSS_LIB_H

#include <vector>
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>

// 生成测试数据（float版本）
inline void generate_test_data(std::vector<std::vector<float>>& A, 
                               std::vector<float>& b, int n) {
    A.assign(n, std::vector<float>(n, 0.0f));
    b.resize(n);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = dis(gen);
                row_sum += std::fabs(A[i][j]);
            }
        }
        A[i][i] = row_sum + std::fabs(dis(gen)) + 1.0f;
        b[i] = dis(gen);
    }
}

// 计算残差
inline float compute_residual(const std::vector<std::vector<float>>& A,
                              const std::vector<float>& x,
                              const std::vector<float>& b) {
    int n = A.size();
    float residual = 0.0f;
    for (int i = 0; i < n; i++) {
        float ax = 0.0f;
        for (int j = 0; j < n; j++) {
            ax += A[i][j] * x[j];
        }
        residual += std::fabs(ax - b[i]);
    }
    return residual;
}

#endif