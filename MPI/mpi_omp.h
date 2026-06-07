#ifndef MPI_OMP_H
#define MPI_OMP_H

#include "common.h"
#include <mpi.h>
#include <vector>
#include <omp.h>

inline void mpi_gaussian_omp(int n, std::vector<float>& x,
                              std::vector<float>& A_orig,
                              std::vector<float>& b_orig) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 0号进程生成数据
    std::vector<float> A, b;
    if (rank == 0) {
        generate_test_data(A, b, n);
        A_orig = A;
        b_orig = b;
    }
    
    // 广播给所有进程
    if (rank != 0) {
        A.resize(n * n);
        b.resize(n);
    }
    MPI_Bcast(A.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // 行块划分
    int rows_per_proc = n / size;
    int rem = n % size;
    int local_n = (rank < rem) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = 0;
    for (int i = 0; i < rank; i++) {
        start_row += (i < rem) ? rows_per_proc + 1 : rows_per_proc;
    }
    
    // 本地存储
    std::vector<float> local_A(local_n * n);
    std::vector<float> local_b(local_n);
    for (int i = 0; i < local_n; i++) {
        int global_row = start_row + i;
        for (int j = 0; j < n; j++) {
            local_A[i * n + j] = A[global_row * n + j];
        }
        local_b[i] = b[global_row];
    }
    
    // 广播缓冲区
    std::vector<float> pivot_row(n);
    float pivot_b;
    
    // 前向消去
    for (int k = 0; k < n; k++) {
        // 确定哪个进程拥有主元行
        int local_has = (start_row <= k && k < start_row + local_n) ? rank : -1;
        int pivot_rank;
        MPI_Allreduce(&local_has, &pivot_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        // 拥有主元行的进程进行归一化
        if (rank == pivot_rank) {
            int local_k = k - start_row;
            float pivot = local_A[local_k * n + k];
            
            for (int j = k; j < n; j++) {
                local_A[local_k * n + j] /= pivot;
            }
            local_b[local_k] /= pivot;
            
            for (int j = 0; j < n; j++) {
                pivot_row[j] = local_A[local_k * n + j];
            }
            pivot_b = local_b[local_k];
        }
        
        MPI_Bcast(pivot_row.data(), n, MPI_FLOAT, pivot_rank, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_FLOAT, pivot_rank, MPI_COMM_WORLD);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_n; i++) {
            int global_row = start_row + i;
            if (global_row > k) {
                float factor = local_A[i * n + k];
                if (factor != 0.0f) {
                    for (int j = k + 1; j < n; j++) {
                        local_A[i * n + j] -= factor * pivot_row[j];
                    }
                    local_b[i] -= factor * pivot_b;
                    local_A[i * n + k] = 0.0f;
                }
            }
        }
    }
    
    // 收集数据到0号进程并回代
    x.resize(n);
    if (rank == 0) {
        std::vector<float> full_A(n * n);
        std::vector<float> full_b(n);
        
        for (int i = 0; i < local_n; i++) {
            int global_row = start_row + i;
            full_b[global_row] = local_b[i];
            for (int j = 0; j < n; j++) {
                full_A[global_row * n + j] = local_A[i * n + j];
            }
        }
        
        int offset = local_n;
        for (int p = 1; p < size; p++) {
            int p_n = (p < rem) ? rows_per_proc + 1 : rows_per_proc;
            std::vector<float> p_A(p_n * n);
            std::vector<float> p_b(p_n);
            MPI_Recv(p_A.data(), p_n * n, MPI_FLOAT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(p_b.data(), p_n, MPI_FLOAT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < p_n; i++) {
                int global_row = offset + i;
                full_b[global_row] = p_b[i];
                for (int j = 0; j < n; j++) {
                    full_A[global_row * n + j] = p_A[i * n + j];
                }
            }
            offset += p_n;
        }
        
        x[n-1] = full_b[n-1] / full_A[(n-1)*n + (n-1)];
        for (int i = n-2; i >= 0; i--) {
            float sum = full_b[i];
            for (int j = i+1; j < n; j++) {
                sum -= full_A[i*n + j] * x[j];
            }
            x[i] = sum / full_A[i*n + i];
        }
        
        for (int p = 1; p < size; p++) {
            MPI_Send(x.data(), n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Send(local_A.data(), local_n * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_b.data(), local_n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(x.data(), n, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

#endif