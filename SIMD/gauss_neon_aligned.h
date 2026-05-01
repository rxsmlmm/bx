//NEON对齐，对消去向量化
#ifndef GAUSS_NEON_ALIGNED_H
#define GAUSS_NEON_ALIGNED_H

#include <vector>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>

// 判断地址是否16字节对齐
inline bool is_aligned_16(const float* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

// 计算从当前列开始，下一个16字节对齐的列索引
inline int next_aligned_j(int j, const std::vector<std::vector<float>>& A, int row) {
    // float是4字节，16字节对齐 = 4个float对齐，即索引是4的倍数
    while (j < (int)A[0].size() && !is_aligned_16(&A[row][j])) {
        j++;
    }
    return j;
}

inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                                 std::vector<float>& b,
                                 std::vector<float>& x) {
    int n = A.size();
    
    //前向消去
    for (int k = 0; k < n; k++) {
        
        // --- 列主元选取 ---
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
            
            float32x4_t v_factor = vdupq_n_f32(factor);
            
            int j = k + 1;
            
            //串行处理到第一个16字节对齐的位置
            int aligned_start = next_aligned_j(j, A, i);
            for (; j < aligned_start && j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            //步骤2：向量化处理对齐的主体部分
            for (; j <= n - 4; j += 4) {
                // 此时地址已对齐，使用对齐加载
                float32x4_t v_aij = vld1q_f32(&A[i][j]);
                float32x4_t v_akj = vld1q_f32(&A[k][j]);
                float32x4_t v_result = vsubq_f32(v_aij, vmulq_f32(v_factor, v_akj));
                vst1q_f32(&A[i][j], v_result);
            }
            
            //串行处理尾部
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            b[i] -= factor * b[k];
        }
    }
    
    //回代求解
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

#endif