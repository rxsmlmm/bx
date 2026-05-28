#ifndef GAUSS_PTHREAD_BAR_H
#define GAUSS_PTHREAD_BAR_H

#include <vector>
#include <pthread.h>
#include <cmath>

struct ThreadParam {
    int t_id;
    int num_threads;
    int n;
    std::vector<std::vector<float>>* A;
    std::vector<float>* b;
    pthread_barrier_t* bar_div;     // 除法完成后同步
    pthread_barrier_t* bar_elim;    // 消去完成后同步
};

void* thread_func(void* arg) {
    ThreadParam* p = (ThreadParam*)arg;
    int t_id = p->t_id;
    int num = p->num_threads;
    int n = p->n;
    auto& A = *(p->A);
    auto& b = *(p->b);
    
    for (int k = 0; k < n; k++) {
        // ===== 除法操作（0 号线程执行）=====
        if (t_id == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[k][j] /= pivot;
            }
            b[k] /= pivot;
            A[k][k] = 1.0f;
        }
        
        // 等待 0 号线程完成除法
        pthread_barrier_wait(p->bar_div);
        
        // ===== 消去操作（所有线程参与，行循环划分）=====
        for (int i = k + 1 + t_id; i < n; i += num) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
            A[i][k] = 0.0f;
        }
        
        // 等待所有线程完成消去
        pthread_barrier_wait(p->bar_elim);
    }
    
    return nullptr;
}

inline void gaussian_elimination_pthread_bar(
    std::vector<std::vector<float>>& A,
    std::vector<float>& b,
    std::vector<float>& x,
    int num_threads)
{
    int n = A.size();
    
    pthread_t* threads = new pthread_t[num_threads];
    ThreadParam* params = new ThreadParam[num_threads];
    
    pthread_barrier_t bar_div, bar_elim;
    pthread_barrier_init(&bar_div, nullptr, num_threads);
    pthread_barrier_init(&bar_elim, nullptr, num_threads);
    
    // 创建工作线程
    for (int i = 0; i < num_threads; i++) {
        params[i].t_id = i;
        params[i].num_threads = num_threads;
        params[i].n = n;
        params[i].A = &A;
        params[i].b = &b;
        params[i].bar_div = &bar_div;
        params[i].bar_elim = &bar_elim;
        pthread_create(&threads[i], nullptr, thread_func, &params[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    pthread_barrier_destroy(&bar_div);
    pthread_barrier_destroy(&bar_elim);
    
    delete[] threads;
    delete[] params;
    
    // 回代（单线程即可）
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