#ifndef GAUSS_COMMON_H
#define GAUSS_COMMON_H

#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include<random>

//生成测试数据
//生成对角占优矩阵 A 和右端向量 b
//A    输出参数：系数矩阵 (n x n)
//B    输出参数：右端向量 (长度 n)
//n    矩阵规模
inline void generate_test_data(std::vector<std::vector<float>>& A, 
                        std::vector<float>& b, 
                        int n) {
    A.assign(n, std::vector<float>(n, 0.0));
    b.resize(n);
    
    //生成随机数，均匀分布在[-10,10]的范围
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        
        // 生成非对角元素
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = dis(gen);
                row_sum += std::fabs(A[i][j]);
            }
        }
        
        // 对角线元素 = 同行非对角元素绝对值之和 + 随机正数 + 1.0
        //这样 |A[i][i]| > sum_|A[i][j]| (i!=j)        
        A[i][i] = row_sum + std::fabs(dis(gen)) + 1.0f;
        
        // 右端向量
        b[i] = dis(gen);
    }
}

//计算残差   ||Ax - b||_1（绝对误差之和） 越小越精确
// A  系数矩阵
// x  解向量
// b  右端向量
inline float compute_residual(const std::vector<std::vector<float>>& A,
                        const std::vector<float>& x,
                        const std::vector<float>& b) {
    int n = A.size();
    float residual = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float ax = 0.0f;
        for (int j = 0; j < n; j++) {
            //计算Ax的第i个分量
            ax += A[i][j] * x[j];
        }
        //累加|Ax-b|
        residual += std::fabs(ax - b[i]);
    }
    
    return residual;
}

//打印矩阵
inline void print_matrix(const std::vector<std::vector<float>>& A, 
                  const std::vector<float>& b,
                  int max_rows = 10) {
    int n = A.size();
    //打印由display_n决定的矩阵中的一部分，观察数据
    int display_n = std::min(n, max_rows);
    
    std::cout << "Matrix A | b:" << std::endl;
    for (int i = 0; i < display_n; i++) {
        for (int j = 0; j < display_n; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << A[i][j] << " ";
        }
        std::cout << "| " << std::setw(8) << b[i] << std::endl;
    }
    if (n > max_rows) {
        std::cout << "... (" << n - max_rows << " more rows)" << std::endl;
    }
    std::cout << std::endl;
}
#endif