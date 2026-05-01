//NEON不对齐，对消去和回代都向量化
#ifndef GAUSS_NEON_FULL_H
#define GAUSS_NEON_FULL_H

#include <vector>
#include <arm_neon.h>
#include <cmath>

inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                                 std::vector<float>& b,
                                 std::vector<float>& x) {
    int n = A.size();
    
    
    // 前向消去  
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
        
        //消去主元下方所有行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / pivot;
            A[i][k] = 0.0f; 
            
            // 将消去因子广播到NEON寄存器的4个通道[factor, factor, factor, factor]
            float32x4_t v_factor = vdupq_n_f32(factor);
            
            int j = k + 1;
            
            //NEON向量化：一次处理4个float
            for (; j <= n - 4; j += 4) {
                // 加载 A[i] 行连续4个元素
                float32x4_t v_aij = vld1q_f32(&A[i][j]);
                
                // 加载 A[k] 行（主元行）连续4个元素
                float32x4_t v_akj = vld1q_f32(&A[k][j]);
                
                // temp = factor * A[k][j..j+3]
                float32x4_t v_temp = vmulq_f32(v_factor, v_akj);
                
                // result = A[i][j..j+3] - temp
                float32x4_t v_result = vsubq_f32(v_aij, v_temp);
                
                // 存回 A[i] 行
                vst1q_f32(&A[i][j], v_result);
            }
            
            //串行处理尾部不足4个的元素
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            // 同步更新右端向量
            b[i] -= factor * b[k];
        }
    }
    
    // 回代求解
    x.resize(n);
    
    // 求解最后一个未知数
    x[n-1] = b[n-1] / A[n-1][n-1];
    
    // 从倒数第二行开始，向上回代
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];           // 主累加器
        float compensation = 0.0f;  // Kahan补偿值
        
        int j = i + 1;
        
        // === NEON向量化：一次并行计算4个 A[i][j] * x[j] ===
        for (; j <= n - 4; j += 4) {
            // 加载 A[i] 行连续4个元素
            float32x4_t v_aij = vld1q_f32(&A[i][j]);
            
            // 加载解向量 x 的连续4个元素
            float32x4_t v_xj = vld1q_f32(&x[j]);
            
            // 并行计算4个乘积：A[i][j] * x[j]
            float32x4_t v_prod = vmulq_f32(v_aij, v_xj);
            
            // 水平求和：把4个乘积加起来,高两个低两个先得到两个和
            float32x2_t v_low  = vget_low_f32(v_prod);   // [p0, p1]
            float32x2_t v_high = vget_high_f32(v_prod);  // [p2, p3]
            float32x2_t v_sum2 = vadd_f32(v_low, v_high); // [p0+p2, p1+p3]
            
            //两个和再相加得到标量和
            float32x2_t v_sum2_dup = vdup_lane_f32(v_sum2, 1); // [p1+p3, p1+p3]
            float32x2_t v_total = vadd_f32(v_sum2, v_sum2_dup); // [p0+p1+p2+p3, ...]
            
            float batch_sum = vget_lane_f32(v_total, 0);
            
            //把4个乘积的和加到累加器
            float y = -batch_sum - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        
        //串行处理剩余元素
        for (; j < n; j++) {
            float y = -A[i][j] * x[j] - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        
        // 除以对角线系数
        x[i] = sum / A[i][i];
    }
}

#endif