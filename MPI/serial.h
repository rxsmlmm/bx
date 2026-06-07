#ifndef SERIAL_H
#define SERIAL_H

#include <vector>

inline void gaussian_elimination(std::vector<float>& A, 
                                  std::vector<float>& b,
                                  std::vector<float>& x,
                                  int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[k * n + k];
        for (int i = k + 1; i < n; i++) {
            float factor = A[i * n + k] / pivot;
            A[i * n + k] = 0.0f;
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    
    x.resize(n);
    x[n - 1] = b[n - 1] / A[(n - 1) * n + (n - 1)];
    for (int i = n - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < n; j++) {
            sum -= A[i * n + j] * x[j];
        }
        x[i] = sum / A[i * n + i];
    }
}

#endif