#include "common.h"
#include "serial.h"
#include "mpi_1d.h"
#include "mpi_1d_col.h"
#include "mpi_pipeline.h"
#include "mpi_simd.h"
#include "mpi_omp.h"
#include "mpi_omp_simd.h"
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <omp.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 解析命令行参数
    if (argc < 6) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <n> <version> <np> <omp> <repeat>" << std::endl;
            std::cerr << "  version: 1=Serial, 2=Row, 3=Col, 4=Pipeline, 5=SIMD, 6=OMP, 7=OMP_SIMD" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int n = std::atoi(argv[1]);
    int version = std::atoi(argv[2]);
    int np = std::atoi(argv[3]);
    int omp_threads = std::atoi(argv[4]);
    int repeat = std::atoi(argv[5]);
    
    // 检查进程数是否匹配
    if (size != np && version != 1) {
        if (rank == 0) {
            std::cerr << "Error: MPI processes (" << size << ") != np (" << np << ")" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "Platform,Version,Size,MPI_Procs,OMP_Threads,Time_ms,Residual,Correct" << std::endl;
    }
    
    double total_time = 0;
    float final_residual = 0;
    bool all_correct = true;
    
    for (int r = 0; r < repeat; r++) {
        double start_time = MPI_Wtime();
        std::vector<float> x;
        float residual = 0.0f;
        bool correct = false;
        
        omp_set_num_threads(omp_threads);
        
        // 串行版本
        if (version == 1) {
            if (rank == 0) {
                std::vector<float> A, b;
                generate_test_data(A, b, n);
                std::vector<float> A_orig = A;
                std::vector<float> b_orig = b;
                
                auto start = std::chrono::high_resolution_clock::now();
                gaussian_elimination(A, b, x, n);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                
                double time_ms = elapsed.count() * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
                
                std::cout << "ARM,Serial," << n << ",1,1," 
                          << std::fixed << std::setprecision(2) << time_ms << ","
                          << residual << "," << (correct ? 1 : 0) << std::endl;
            }
        }
        else if (version == 2) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_1d(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
        else if (version == 3) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_1d_col(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
        else if (version == 4) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_pipeline(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
        else if (version == 5) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_simd(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
        else if (version == 6) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_omp(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
        else if (version == 7) {
            std::vector<float> A_orig, b_orig;
            mpi_gaussian_omp_simd(n, x, A_orig, b_orig);
            if (rank == 0) {
                residual = compute_residual(A_orig, x, b_orig, n);
                correct = (residual < 1e-2f);
                double time_ms = (MPI_Wtime() - start_time) * 1000.0;
                total_time += time_ms;
                final_residual = residual;
                if (!correct) all_correct = false;
            }
        }
    }
    
    if (rank == 0 && all_correct) {
        std::cout << "Average: " << total_time / repeat << " ms" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}