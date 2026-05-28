#ifndef GAUSS_PTHREAD_BAR_SIMD_H
#define GAUSS_PTHREAD_BAR_SIMD_H

#include <vector>
#include <pthread.h>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#else
#include <xmmintrin.h>
#endif

struct SIMDParam {
    int t_id;
    int num_threads;
    int n;
    std::vector<std::vector<float>>* A;
    std::vector<float>* b;
    pthread_barrier_t* bar_div;
    pthread_barrier_t* bar_elim;
};

void* simd_thread_func(void* arg) {
    SIMDParam* p = (SIMDParam*)arg;
    int t_id = p->t_id;
    int num = p->num_threads;
    int n = p->n;
    auto& A = *(p->A);
    auto& b = *(p->b);
    
    for (int k = 0; k < n; k++) {
        // 除法
        if (t_id == 0) {
            float pivot = A[k][k];
            float inv_pivot = 1.0f / pivot;
            
#ifdef __aarch64__
            float32x4_t vinv = vdupq_n_f32(inv_pivot);
            int j;
            for (j = k + 1; j + 3 < n; j += 4) {
                float32x4_t v = vld1q_f32(&A[k][j]);
                v = vmulq_f32(v, vinv);
                vst1q_f32(&A[k][j], v);
            }
            for (; j < n; j++) A[k][j] *= inv_pivot;
#else
            __m128 vinv = _mm_set1_ps(inv_pivot);
            int j;
            for (j = k + 1; j + 3 < n; j += 4) {
                __m128 v = _mm_loadu_ps(&A[k][j]);
                v = _mm_mul_ps(v, vinv);
                _mm_storeu_ps(&A[k][j], v);
            }
            for (; j < n; j++) A[k][j] *= inv_pivot;
#endif
            b[k] *= inv_pivot;
            A[k][k] = 1.0f;
        }
        pthread_barrier_wait(p->bar_div);
        
        // 消去
        for (int i = k + 1 + t_id; i < n; i += num) {
            float factor = A[i][k];
#ifdef __aarch64__
            float32x4_t vfactor = vdupq_n_f32(factor);
            int j;
            for (j = k + 1; j + 3 < n; j += 4) {
                float32x4_t va = vld1q_f32(&A[k][j]);
                float32x4_t vb = vld1q_f32(&A[i][j]);
                vb = vmlsq_f32(vb, vfactor, va);
                vst1q_f32(&A[i][j], vb);
            }
#else
            __m128 vfactor = _mm_set1_ps(factor);
            int j;
            for (j = k + 1; j + 3 < n; j += 4) {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                __m128 vb = _mm_loadu_ps(&A[i][j]);
                vb = _mm_sub_ps(vb, _mm_mul_ps(vfactor, va));
                _mm_storeu_ps(&A[i][j], vb);
            }
#endif
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
            A[i][k] = 0.0f;
        }
        pthread_barrier_wait(p->bar_elim);
    }
    return nullptr;
}

inline void gaussian_elimination_pthread_bar_simd(
    std::vector<std::vector<float>>& A,
    std::vector<float>& b,
    std::vector<float>& x,
    int num_threads)
{
    int n = A.size();
    
    pthread_t* threads = new pthread_t[num_threads];
    SIMDParam* params = new SIMDParam[num_threads];
    
    pthread_barrier_t bar_div, bar_elim;
    pthread_barrier_init(&bar_div, nullptr, num_threads);
    pthread_barrier_init(&bar_elim, nullptr, num_threads);
    
    for (int i = 0; i < num_threads; i++) {
        params[i].t_id = i;
        params[i].num_threads = num_threads;
        params[i].n = n;
        params[i].A = &A;
        params[i].b = &b;
        params[i].bar_div = &bar_div;
        params[i].bar_elim = &bar_elim;
        pthread_create(&threads[i], nullptr, simd_thread_func, &params[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    pthread_barrier_destroy(&bar_div);
    pthread_barrier_destroy(&bar_elim);
    
    delete[] threads;
    delete[] params;
    
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