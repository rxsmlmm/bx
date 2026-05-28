#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "gauss_common.h"
#include "gauss_serial.h"
#include "gauss_pthread_bar.h"
#include "gauss_omp.h"
#include "gauss_omp_simd.h"
#include "gauss_pthread_dyn.h"

int main(int argc, char* argv[]) {
    const char* ver = argv[1];
    int n = atoi(argv[2]);
    int threads = (argc >= 4) ? atoi(argv[3]) : 1;
    
    std::vector<std::vector<float>> A;
    std::vector<float> b, x;
    generate_test_data(A, b, n);
    
    if (strcmp(ver, "serial") == 0)
        gaussian_elimination(A, b, x);
    else if (strcmp(ver, "pthread_bar") == 0)
        gaussian_elimination_pthread_bar(A, b, x, threads);
    else if (strcmp(ver, "omp") == 0)
        gaussian_elimination_omp(A, b, x, threads);
    else if (strcmp(ver, "omp_simd") == 0)
        gaussian_elimination_omp_simd(A, b, x, threads);
    else if (strcmp(ver, "pthread_dyn") == 0)
    gaussian_elimination_pthread_dyn(A, b, x, threads);
    
    std::cout << "done" << std::endl;
    return 0;
}


