#ifndef GAUSS_OMP_H
#define GAUSS_OMP_H

#include <vector>
#include <cmath>
#include <omp.h>

// SCHEDULE 可选: static, dynamic, guided
#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE static
#endif

inline void gaussian_elimination_omp(
    std::vector<std::vector<float>>& A,
    std::vector<float>& b,
    std::vector<float>& x,
    int num_threads)
{
    int n = A.size();
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        for (int k = 0; k < n; k++) {
            // 除法：单线程执行
            #pragma omp single
            {
                float pivot = A[k][k];
                for (int j = k + 1; j < n; j++) {
                    A[k][j] /= pivot;
                }
                b[k] /= pivot;
                A[k][k] = 1.0f;
            }
            
            // 消去：多线程并行，按行划分
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                float factor = A[i][k];
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
                A[i][k] = 0.0f;
            }
            // 隐式 barrier，所有线程同步后再进入下一轮 k
        }
    }
    
    // 回代
    x.resize(n);
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}

#endif