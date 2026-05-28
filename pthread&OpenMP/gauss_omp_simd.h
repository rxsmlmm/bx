#ifndef GAUSS_OMP_SIMD_H
#define GAUSS_OMP_SIMD_H

#include <vector>
#include <cmath>
#include <omp.h>

inline void gaussian_elimination_omp_simd(
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
            #pragma omp single
            {
                float pivot = A[k][k];
                float inv = 1.0f / pivot;
                #pragma omp simd
                for (int j = k + 1; j < n; j++) {
                    A[k][j] *= inv;
                }
                b[k] *= inv;
                A[k][k] = 1.0f;
            }
            
            #pragma omp for schedule(static)
            for (int i = k + 1; i < n; i++) {
                float factor = A[i][k];
                #pragma omp simd
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
                A[i][k] = 0.0f;
            }
        }
    }
    
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