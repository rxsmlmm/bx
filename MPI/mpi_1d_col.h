#ifndef MPI_1D_COL_H
#define MPI_1D_COL_H

#include "common.h"
#include <mpi.h>
#include <vector>

inline void mpi_gaussian_1d_col(int n, std::vector<float>& x,
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
    
    if (rank != 0) {
        A.resize(n * n);
        b.resize(n);
    }
    MPI_Bcast(A.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // 列块划分
    int cols_per_proc = n / size;
    int rem = n % size;
    int local_cols = (rank < rem) ? cols_per_proc + 1 : cols_per_proc;
    int start_col = 0;
    for (int i = 0; i < rank; i++) {
        start_col += (i < rem) ? cols_per_proc + 1 : cols_per_proc;
    }
    
    // 每个进程存储完整的n行，但只负责local_cols列
    std::vector<float> local_A(n * local_cols);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < local_cols; j++) {
            local_A[i * local_cols + j] = A[i * n + (start_col + j)];
        }
    }
    // b向量每个进程都存完整副本，独立更新
    std::vector<float> local_b = b;
    
    // 准备Allgatherv的参数
    std::vector<int> col_counts(size), col_displs(size);
    int tmp = 0;
    for (int p = 0; p < size; p++) {
        int p_cols = (p < rem) ? cols_per_proc + 1 : cols_per_proc;
        col_counts[p] = p_cols;
        col_displs[p] = tmp;
        tmp += p_cols;
    }
    
    // 前向消去
    for (int k = 0; k < n; k++) {
        // 找到拥有第k列的进程
        int pivot_owner = -1;
        int offset = 0;
        for (int p = 0; p < size; p++) {
            int p_cols = (p < rem) ? cols_per_proc + 1 : cols_per_proc;
            if (offset <= k && k < offset + p_cols) {
                pivot_owner = p;
                break;
            }
            offset += p_cols;
        }
        
        float pivot = 0.0f;
        if (rank == pivot_owner) {
            pivot = local_A[k * local_cols + (k - start_col)];
        }
        MPI_Bcast(&pivot, 1, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
        
        for (int j = 0; j < local_cols; j++) {
            int global_col = start_col + j;
            if (global_col >= k) {
                local_A[k * local_cols + j] /= pivot;
            }
        }
        local_b[k] /= pivot;
        
        // 收集完整的归一化第k行
        std::vector<float> full_row(n);
        MPI_Allgatherv(local_A.data() + k * local_cols, local_cols, MPI_FLOAT,
                       full_row.data(), col_counts.data(), col_displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        
        // 所有进程对自己负责的行做消去
        for (int i = k + 1; i < n; i++) {
            float factor = 0.0f;
            if (rank == pivot_owner) {
                factor = local_A[i * local_cols + (k - start_col)];
            }
            MPI_Bcast(&factor, 1, MPI_FLOAT, pivot_owner, MPI_COMM_WORLD);
            
            if (factor != 0.0f) {
                for (int j = 0; j < local_cols; j++) {
                    int global_col = start_col + j;
                    if (global_col > k) {
                        local_A[i * local_cols + j] -= factor * full_row[global_col];
                    } else if (global_col == k) {
                        local_A[i * local_cols + j] = 0.0f;
                    }
                }
                local_b[i] -= factor * local_b[k];
            }
        }
    }
    
    // 收集完整的上三角矩阵
    std::vector<float> full_A(n * n);
    for (int i = 0; i < n; i++) {
        MPI_Allgatherv(local_A.data() + i * local_cols, local_cols, MPI_FLOAT,
                       full_A.data() + i * n, col_counts.data(), col_displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
    }
    
    x.resize(n);
    if (rank == 0) {
        x[n-1] = local_b[n-1] / full_A[(n-1)*n + (n-1)];
        for (int i = n-2; i >= 0; i--) {
            float sum = local_b[i];
            for (int j = i+1; j < n; j++) {
                sum -= full_A[i*n + j] * x[j];
            }
            x[i] = sum / full_A[i*n + i];
        }
        
        for (int p = 1; p < size; p++) {
            MPI_Send(x.data(), n, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(x.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

#endif