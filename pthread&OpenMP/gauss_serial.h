#ifndef GAUSS_SERIAL_H
#define GAUSS_SERIAL_H

#include <vector>

//串行高斯消去法求解 Ax = b
// A  系数矩阵（会被修改）
// b  右端向量（会被修改）
// x  输出：解向量
inline void gaussian_elimination(std::vector<std::vector<float>>& A, 
                          std::vector<float>& b,
                          std::vector<float>& x) {
    int n = A.size();
    
    //前向消去
    for (int k = 0; k < n; k++) {
        float pivot = A[k][k];
        
        // 对主元下方的每一行
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k] / pivot;
            A[i][k]=0.0;
            // 更新第i行的第 k+1 到 n-1 列
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            
            // 更新右端向量
            b[i] -= factor * b[k];
        }
    }
    
    //回代求解
    x.resize(n);
    x[n-1] = b[n-1] / A[n-1][n-1];
    //从倒数第二行向上，减去已知变量的贡献，再除以对角线系数
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}
#endif