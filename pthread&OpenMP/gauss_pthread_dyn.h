#ifndef GAUSS_PTHREAD_DYN_H
#define GAUSS_PTHREAD_DYN_H

#include <vector>
#include <pthread.h>
#include <cmath>

struct DynParam {
    int t_id;
    int k;
    int n;
    int num_threads;
    std::vector<std::vector<float>>* A;
    std::vector<float>* b;
};

void* dyn_thread_func(void* arg) {
    DynParam* p = (DynParam*)arg;
    int t_id = p->t_id;
    int k = p->k;
    int n = p->n;
    int num = p->num_threads;
    auto& A = *(p->A);
    auto& b = *(p->b);
    
    for (int i = k + 1 + t_id; i < n; i += num) {
        float factor = A[i][k];
        for (int j = k + 1; j < n; j++) {
            A[i][j] -= factor * A[k][j];
        }
        b[i] -= factor * b[k];
        A[i][k] = 0.0f;
    }
    
    return nullptr;
}

inline void gaussian_elimination_pthread_dyn(
    std::vector<std::vector<float>>& A,
    std::vector<float>& b,
    std::vector<float>& x,
    int num_threads)
{
    int n = A.size();
    
    for (int k = 0; k < n; k++) {
        // 除法
        float pivot = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
        }
        b[k] /= pivot;
        A[k][k] = 1.0f;
        
        // 动态创建线程做消去
        // 消去：主线程处理 t_id=0 的部分，创建 num_threads-1 个子线程
        int worker_count = num_threads - 1;
        pthread_t* threads = new pthread_t[worker_count];
        DynParam* params = new DynParam[worker_count];

        for (int i = 0; i < worker_count; i++) {
           params[i].t_id = i + 1;  // 子线程从 1 开始
           params[i].k = k;
           params[i].n = n;
           params[i].num_threads = num_threads;
           params[i].A = &A;
           params[i].b = &b;
           pthread_create(&threads[i], nullptr, dyn_thread_func, &params[i]);
       }       

       // 主线程做 t_id=0 的消去
       for (int i = k + 1; i < n; i += num_threads) {
         float factor = A[i][k];
         for (int j = k + 1; j < n; j++) {
             A[i][j] -= factor * A[k][j];
         }  
         b[i] -= factor * b[k];
         A[i][k] = 0.0f;
       } 

       for (int i = 0; i < worker_count; i++) {
         pthread_join(threads[i], nullptr);
       }
        
        delete[] threads;
        delete[] params;
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