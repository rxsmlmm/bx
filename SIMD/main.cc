#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include<random>
#include "gauss_common.h"
//#include <string>
//#include <fstream>
//#include <sstream>
//#include <sys/time.h>
//#include <omp.h>
// 自行添加需要的头文件
// TODO : 完成你的高斯消元
// 如果需要文件输入输出, 请使用相对路径并放在 files 文件夹下

//通过注释切换版本
#define SERIAL
//#define NEON_UNALIGNED
//#define NEON_ALIGNED
//#define NEON_FULL
//#define NEON_UNROLL

#ifdef SERIAL
   #include "gauss_serial.h"
#elif defined(NEON_UNALIGNED)
   #include "gauss_neon_unaligned.h"
#elif defined(NEON_ALIGNED)
   #include "gauss_neon_aligned.h"
#elif defined(NEON_FULL)
   #include "gauss_neon_full.h"
#elif defined(NEON_UNROLL)
   #include "gauss_neon_unroll.h"
#endif




int main(int argc, char *argv[])
{
      // 配置参数
    const int n = 1024;          // 矩阵规模，可以修改测不同的规模
    const bool verbose = false;   // 是否打印详细信息  
    
std::cout << "Version: ";
#ifdef SERIAL
    std::cout << "SERIAL";
#elif defined(NEON_UNALIGNED)
    std::cout << "NEON_UNALIGNED";
#elif defined(NEON_ALIGNED)
    std::cout << "NEON_ALIGNED";
#elif defined(NEON_FULL)
    std::cout << "NEON_FULL";
#elif defined(NEON_UNROLL)
    std::cout << "NEON_UNROLL";
#endif
    std::cout << std::endl;
    
    std::cout << "Matrix size: " << n << " x " << n << std::endl;

    
    // 生成测试数据
    std::vector<std::vector<float>> A;
    std::vector<float> b, x;
    generate_test_data(A, b, n);
    
    if (verbose) {
        std::cout << "\\n[Original Matrix]" << std::endl;
        print_matrix(A, b, 5);
    }
    
    //保存原始矩阵和右端项，用来计算残差
    std::vector<std::vector<float>> A_original = A;
    std::vector<float> b_original = b;
    // =================================

    // 计时并执行高斯消去
    auto Start = std::chrono::high_resolution_clock::now();
    gaussian_elimination(A, b, x);
    auto End = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;

    //验证结果
    float residual = compute_residual(A_original, x, b_original);  
    
    if (verbose && n <= 20) {
        std::cout << "\\n[Solution x]" << std::endl;
        for (int i = 0; i < std::min(n, 10); i++) {
            std::cout << "x[" << i << "] = " << std::setw(12) 
                      << std::fixed << std::setprecision(6) << x[i] << std::endl;
        }
    }
    
    // 输出结果
    std::cout << "\\n[Results]" << std::endl;
    std::cout<<"average latency  : "<<elapsed.count()<<" (ms) "<<std::endl;
    std::cout << "Residual: " << std::scientific << std::setprecision(6) 
              << residual << std::endl;
    
    // 判断正确性
    if (residual < 1e-2f) {
        std::cout << "Status:   PASS (residual < 1e-2f)" << std::endl;
    } else {
        std::cout << "Status:   WARNING (residual too large)" << std::endl;
    }
    return 0;
}
