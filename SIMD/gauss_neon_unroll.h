//NEON两倍展开
#ifndef GAUSS_NEON_UNROLL_H
#define GAUSS_NEON_UNROLL_H

#include <vector>
#include <arm_neon.h>
#include <cmath>

inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                                 std::vector<float>& b,
                                 std::vector<float>& x) {
    int n = A.size();
    
    //前向消去
    for (int k = 0; k < n; k++) {
        
        //列主元选取：找到第k列绝对值最大的行
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
        
        //消去主元下方所有行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / pivot;
            A[i][k] = 0.0f;
            
            // 将factor广播到4个通道
            float32x4_t v_factor = vdupq_n_f32(factor);
            
            int j = k + 1;
            
            //NEON 2倍循环展开：一次处理8个float（两个NEON向量
            for (; j <= n - 8; j += 8) {
                // --- 第一批4个：A[i][j..j+3] ---
                float32x4_t v_aij_0 = vld1q_f32(&A[i][j]);
                float32x4_t v_akj_0 = vld1q_f32(&A[k][j]);
                float32x4_t v_result_0 = vsubq_f32(v_aij_0, vmulq_f32(v_factor, v_akj_0));
                vst1q_f32(&A[i][j], v_result_0);
                
                // --- 第二批4个：A[i][j+4..j+7] ---
                float32x4_t v_aij_1 = vld1q_f32(&A[i][j + 4]);
                float32x4_t v_akj_1 = vld1q_f32(&A[k][j + 4]);
                float32x4_t v_result_1 = vsubq_f32(v_aij_1, vmulq_f32(v_factor, v_akj_1));
                vst1q_f32(&A[i][j + 4], v_result_1);
            }
            
            // 处理剩余能用向量化的部分（不足8个但≥4个）
            // 条件 j <= n-4：保证还能一次处理4个
            for (; j <= n - 4; j += 4) {
                float32x4_t v_aij = vld1q_f32(&A[i][j]);
                float32x4_t v_akj = vld1q_f32(&A[k][j]);
                float32x4_t v_result = vsubq_f32(v_aij, vmulq_f32(v_factor, v_akj));
                vst1q_f32(&A[i][j], v_result);
            }
            
            //串行处理尾部不足4个的元素
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            // 更新右端向量
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