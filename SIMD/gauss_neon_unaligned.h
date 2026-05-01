//NEON不对齐，对消去向量化
#ifndef GAUSS_NEON_UNALIGNED_H
#define GAUSS_NEON_UNALIGNED_H

#include <vector>
#include <arm_neon.h>
#include <cmath>

inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                                 std::vector<float>& b,
                                 std::vector<float>& x) {
    int n = A.size();
    
    //前向消去
    for (int k = 0; k < n; k++) {

        //找到第k列绝对值最大的行
        int max_row = k;
        float max_val = std::fabs(A[k][k]);
        for (int r = k + 1; r < n; r++) {
            float val = std::fabs(A[r][k]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }
        
        // 如果主元不在当前行，交换两行
        if (max_row != k) {
            std::swap(A[k], A[max_row]);
            std::swap(b[k], b[max_row]);
        }

        float pivot = A[k][k];
        
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / pivot;
            A[i][k] = 0.0f;
            
            // 将factor广播到NEON寄存器的4个通道
            float32x4_t v_factor = vdupq_n_f32(factor);
            
            int j = k + 1;
            
            //NEON向量化部分：一次处理4个float
            for (; j <= n - 4; j += 4) {
                // 加载 A[i][j..j+3]
                float32x4_t v_aij = vld1q_f32(&A[i][j]);
                
                // 加载 A[k][j..j+3]
                float32x4_t v_akj = vld1q_f32(&A[k][j]);
                
                // 计算：v_temp = factor * v_akj
                float32x4_t v_temp = vmulq_f32(v_factor, v_akj);
                
                // 计算：v_result = v_aij - v_temp
                float32x4_t v_result = vsubq_f32(v_aij, v_temp);
                
                // 存回 A[i][j..j+3]
                vst1q_f32(&A[i][j], v_result);
            }
            
            //串行处理剩余元素
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
        float compensation = 0.0f;  // 记录丢失的精度
        
        for (int j = i + 1; j < n; j++) {
            // Kahan补偿：每次加法修正上次丢失的精度
            float y = -A[i][j] * x[j] - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;  // 计算本次丢失的精度
            sum = t;
        }
        
        x[i] = sum / A[i][i];
    }
}


#endif