#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <xmmintrin.h>
#include "gauss_lib.h"

#define SSE  // SSE向量化版本

// ==================== SSE向量化高斯消去 ====================
inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                                 std::vector<float>& b,
                                 std::vector<float>& x) {
    int n = A.size();
    
    // 阶段1：前向消去（SSE 4路向量化 + 列主元）
    for (int k = 0; k < n; k++) {
        // 列主元选取
        int max_row = k;
        float max_val = std::fabs(A[k][k]);
        for (int r = k + 1; r < n; r++) {
            float val = std::fabs(A[r][k]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(b[k], b[max_row]);
        }
        
        float pivot = A[k][k];
        
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / pivot;
            A[i][k] = 0.0f;
            
            // 广播factor到SSE寄存器4个通道
            __m128 v_factor = _mm_set1_ps(factor);
            
            int j = k + 1;
            
            // SSE向量化：一次处理4个float
            for (; j <= n - 4; j += 4) {
                __m128 v_aij = _mm_loadu_ps(&A[i][j]);
                __m128 v_akj = _mm_loadu_ps(&A[k][j]);
                __m128 v_temp = _mm_mul_ps(v_factor, v_akj);
                __m128 v_result = _mm_sub_ps(v_aij, v_temp);
                _mm_storeu_ps(&A[i][j], v_result);
            }
            
            // 串行处理尾部
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            b[i] -= factor * b[k];
        }
    }
    
    // 阶段2：回代求解（Kahan补偿）
    x.resize(n);
    x[n-1] = b[n-1] / A[n-1][n-1];
    
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        float compensation = 0.0f;
        for (int j = i + 1; j < n; j++) {
            float y = -A[i][j] * x[j] - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        x[i] = sum / A[i][i];
    }
}

int main() {
    const int n = 1024;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Version: SSE (x86, 4-way vectorization)" << std::endl;
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<std::vector<float>> A;
    std::vector<float> b, x;
    generate_test_data(A, b, n);
    
    // 保存原始数据用于验证
    std::vector<std::vector<float>> A_original = A;
    std::vector<float> b_original = b;
    
    auto Start = std::chrono::high_resolution_clock::now();
    gaussian_elimination(A, b, x);
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = End - Start;
    
    float residual = compute_residual(A_original, x, b_original);
    
    std::cout << "\\n[Results]" << std::endl;
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Residual: " << std::scientific << std::setprecision(6) << residual << std::endl;
    
    if (residual < 1e-2f) {
        std::cout << "Status: PASS (float tolerance 1e-2)" << std::endl;
    } else {
        std::cout << "Status: WARNING" << std::endl;
    }
    
    return 0;
}