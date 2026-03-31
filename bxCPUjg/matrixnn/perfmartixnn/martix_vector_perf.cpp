#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
using namespace std;
using namespace std::chrono;

// Linux计时类
class counttime {
private:
    high_resolution_clock::time_point m_start;
    high_resolution_clock::time_point m_end;
public:
    void start() { m_start = high_resolution_clock::now(); }
    void stop() { m_end = high_resolution_clock::now(); }
    double getelapsedmilliseconds() const {
        return duration_cast<microseconds>(m_end - m_start).count() / 1000.0;
    }
};

// 生成测试的矩阵规模（结合你的CPU缓存大小）
vector<int> generatetestmatrix() {
    vector<int> size;
    // L1缓存范围: 512KB，每个int 4字节，矩阵大小约 512KB/4 ≈ 128K个元素，规模约√128K ≈ 358
    for (int n = 32; n <= 256; n += 32) {
        size.push_back(n);  // 观察L1缓存内行为
    }
    // L2缓存范围: 8MB，约 8MB/4 ≈ 2M个元素，规模约√2M ≈ 1414
    for (int n = 384; n <= 1024; n += 128) {
        size.push_back(n);  // 观察L2缓存内行为
    }
    // L3缓存范围: 16MB，约 16MB/4 ≈ 4M个元素，规模约√4M ≈ 2000
    for (int n = 1280; n <= 4096; n += 256) {
        size.push_back(n);  // 观察L3缓存行为及超出
    }
    return size;
}

// 根据矩阵规模制定重复次数
int getrepeat(int n) {
    if (n <= 128) return 500;
    if (n <= 256) return 200;
    if (n <= 512) return 100;
    if (n <= 1024) return 50;
    if (n <= 2048) return 20;
    return 10;
}

// 生成测试数据
void generatetestdata(vector<vector<int>>& matrix, vector<int>& vec, int n) {
    matrix.resize(n);
    for (int i = 0; i < n; i++) {
        matrix[i].resize(n);
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i + j;
        }
    }
    vec.resize(n);
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
}

// 平凡算法：列优先访问
vector<int> ordinary(const vector<vector<int>>& matrix,
    const vector<int>& vec, int n) {
    vector<int> result(n, 0);
    for (int j = 0; j < n; j++) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += matrix[i][j] * vec[i];
        }
        result[j] = sum;
    }
    return result;
}

// cache优化算法：行优先访问
vector<int> cache(const vector<vector<int>>& matrix,
    const vector<int>& vec, int n) {
    vector<int> result(n, 0);
    for (int i = 0; i < n; i++) {
        int vi = vec[i];
        const vector<int>& row = matrix[i];
        for (int j = 0; j < n; j++) {
            result[j] += row[j] * vi;
        }
    }
    return result;
}

// 验证结果
bool compareresults(const vector<int>& result1, const vector<int>& result2, int n) {
    for (int i = 0; i < n; i++) {
        if (result1[i] != result2[i]) {
            return false;
        }
    }
    return true;
}

// 测试结果结构
struct testresult {
    int size;
    double ordinarytime;
    double cachetime;
    double speedup;
    double matrixsizemb;
    bool correct;
};

// 测试单个规模
testresult test(int n) {
    testresult result;
    result.size = n;
    result.matrixsizemb = (n * n * sizeof(int)) / (1024.0 * 1024.0);

    vector<vector<int>> matrix;
    vector<int> vec;
    generatetestdata(matrix, vec, n);
    int repeatcount = getrepeat(n);

    // 测试平凡算法
    counttime time1;
    double totaltime1 = 0.0;
    vector<int> result1;

    for (int i = 0; i < repeatcount; i++) {
        time1.start();
        result1 = ordinary(matrix, vec, n);
        time1.stop();
        totaltime1 += time1.getelapsedmilliseconds();
    }
    result.ordinarytime = totaltime1 / repeatcount;

    // 测试优化算法
    counttime time2;
    double totaltime2 = 0.0;
    vector<int> result2;

    for (int i = 0; i < repeatcount; i++) {
        time2.start();
        result2 = cache(matrix, vec, n);
        time2.stop();
        totaltime2 += time2.getelapsedmilliseconds();
    }
    result.cachetime = totaltime2 / repeatcount;

    result.correct = compareresults(result1, result2, n);
    result.speedup = result.ordinarytime / result.cachetime;

    return result;
}

// 输出结果表格
void printresult(const vector<testresult>& results) {
    cout << "\\n" << string(100, '=') << endl;
    cout << left << setw(10) << "规模(n)"
        << setw(15) << "矩阵大小(MB)"
        << setw(18) << "平凡算法(ms)"
        << setw(18) << "优化算法(ms)"
        << setw(12) << "加速比"
        << "状态" << endl;
    cout << string(100, '-') << endl;

    cout << fixed << setprecision(3);
    for (const auto& r : results) {
        cout << left << setw(10) << r.size
            << setw(15) << r.matrixsizemb
            << setw(18) << r.ordinarytime
            << setw(18) << r.cachetime
            << setw(12) << r.speedup
            << (r.correct ? "正确" : "错误") << endl;
    }
    cout << string(100, '=') << endl;
}

// 导出CSV
void ExportToCSV(const vector<testresult>& results, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法创建文件: " << filename << endl;
        return;
    }

    file << "matrix_size,matrix_size_mb,ordinary_time_ms,cache_time_ms,speedup\\n";
    file << fixed << setprecision(6);
    for (const auto& r : results) {
        file << r.size << ","
            << r.matrixsizemb << ","
            << r.ordinarytime << ","
            << r.cachetime << ","
            << r.speedup << "\\n";
    }
    file.close();
    cout << "\\n数据已导出到: " << filename << endl;
}

int main() {
    cout << "=== 矩阵-向量内积性能测试 ===" << endl;
    cout << "CPU: AMD Ryzen 7 8845H" << endl;
    cout << "L1缓存: 512KB | L2缓存: 8MB | L3缓存: 16MB\\n" << endl;

    vector<int> testSizes = generatetestmatrix();
    cout << "测试规模数量: " << testSizes.size() << endl;
    cout << "范围: " << testSizes.front() << " ~ " << testSizes.back() << "\\n" << endl;

    vector<testresult> results;
    int current = 0;

    cout << "开始性能测试..." << endl;
    for (int n : testSizes) {
        current++;
        cout << "[" << current << "/" << testSizes.size() << "] 测试规模 " << n << " ... " << flush;

        testresult r = test(n);
        results.push_back(r);

        cout << "平凡: " << fixed << setprecision(3) << r.ordinarytime
            << " ms, 优化: " << r.cachetime << " ms, 加速比: "
            << setprecision(2) << r.speedup << "x" << endl;
    }

    printresult(results);
    ExportToCSV(results, "performance_data.csv");

    cout << "\\n测试完成！" << endl;
    return 0;
}
