#include <iostream>
#include <vector>
#include <windows.h>
#include <iomanip>
#include <algorithm>
using namespace std;

// 平凡算法：逐个累加
double naive_sum(const vector<double>& arr) {
    double sum = 0.0;
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// 两路链式累加
double two_way_sum(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0;
    size_t n = arr.size();
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
    }
    for (; i < n; ++i) {
        sum1 += arr[i];
    }
    return sum1 + sum2;
}

// 四路链式累加
double four_way_sum(const vector<double>& arr) {
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    size_t n = arr.size();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
        sum3 += arr[i + 2];
        sum4 += arr[i + 3];
    }
    for (; i < n; ++i) {
        sum1 += arr[i];
    }
    return sum1 + sum2 + sum3 + sum4;
}

// Windows高精度计时
template<typename Func>
double measure_time(Func func, const vector<double>& arr, int repeat = 10) {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    volatile double result = 0.0;
    for (int i = 0; i < repeat; ++i) {
        result = func(arr);
    }
    QueryPerformanceCounter(&end);

    double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;
    return elapsed / repeat;
}

int main() {
    cout << "========== n个数求和性能测试 ==========" << endl;

    const size_t bytes_per_element = sizeof(double);

    // 生成20多组测试规模，覆盖各级缓存
    vector<size_t> test_sizes = {
        // 小规模
        10000, 20000, 50000, 80000, 100000,
        // L2边界附近 (512KB -> 65536个元素)
        150000, 200000, 250000, 300000, 350000, 400000,
        // L3 8MB边界附近 (1048576个元素)
        500000, 600000, 700000, 800000, 900000,
        1000000, 1100000, 1200000, 1300000, 1400000,
        // L3 16MB边界附近 (2097152个元素)
        1500000, 1600000, 1700000, 1800000, 1900000, 2000000,
        2100000, 2200000, 2300000, 2400000, 2500000,
        // 更大规模
        3000000, 3500000, 4000000, 5000000, 6000000, 8000000,
        10000000, 12000000, 15000000, 20000000
    };

    // 去重排序
    sort(test_sizes.begin(), test_sizes.end());
    test_sizes.erase(unique(test_sizes.begin(), test_sizes.end()), test_sizes.end());

    const int repeat_count = 10;

    // 输出表头
    cout << left << setw(15) << "规模(N)"
        << setw(15) << "数据大小(MB)"
        << setw(18) << "平凡算法(ms)"
        << setw(18) << "两路链式(ms)"
        << setw(18) << "四路链式(ms)" << endl;
    cout << string(80, '-') << endl;

    for (size_t n : test_sizes) {
        // 生成测试数据
        vector<double> arr(n);
        for (size_t i = 0; i < n; ++i) {
            arr[i] = static_cast<double>(i) * 0.1;
        }

        double data_mb = static_cast<double>(n * bytes_per_element) / (1024 * 1024);

        // 测量时间
        double naive_time = measure_time(naive_sum, arr, repeat_count) * 1000;
        double two_way_time = measure_time(two_way_sum, arr, repeat_count) * 1000;
        double four_way_time = measure_time(four_way_sum, arr, repeat_count) * 1000;

        // 输出
        cout << left << setw(15) << n
            << setw(15) << fixed << setprecision(2) << data_mb
            << setw(18) << fixed << setprecision(6) << naive_time
            << setw(18) << two_way_time
            << setw(18) << four_way_time << endl;
    }

    cout << "\\n测试完成。" << endl;
    return 0;
}