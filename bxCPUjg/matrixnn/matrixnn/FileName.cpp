//矩阵与向量内积

//CPU: AMD Ryzen 7 8845H
//L1缓存: 512KB
//L2缓存: 8MB
//L3缓存: 16MB

#include <iostream>
#include <vector>
#include <iomanip>
#include <windows.h>
#include <fstream>
using namespace std;

//生成测试的矩阵
vector<int> generatetestmatrix() {
    vector<int> size;
    //小规模：观察L1缓存
    for (int n = 32; n <= 256; n += 32) {
        size.push_back(n);
    }
    //中规模：观察L2缓存
    for (int n = 384; n <= 1024; n += 128) {
        size.push_back(n);
    }
    // 大规模：观察L3缓存及超出
    for (int n = 1280; n <= 4096; n += 256) {
        size.push_back(n);
    }
    return size;
}

//根据矩阵的规模制定重复的次数
int getrepeat(int n) {
    if (n <= 128) return 500;      // 极小规模重复500次
    if (n <= 256) return 200;      // 小规模重复200次
    if (n <= 512) return 100;      // 中小规模重复100次
    if (n <= 1024) return 50;      // 中等规模重复50次
    if (n <= 2048) return 20;      // 大规模重复20次
    return 10;                      // 超大规模重复10次
}

//生成测试的数据，矩阵用i+j，向量用i
void generatetestdata(vector<vector<int>>& matrix, vector<int>& vec, int n) {
    matrix.resize(n);
    for (int i = 0; i < n; i++) {
        matrix[i].resize(n);
        for (int j = 0; j < n; j++) {
            matrix[i][j] =i + j;
        }
    }
    vec.resize(n);
    for (int i = 0; i < n; i++) {
        vec[i] = i;
    }
}

//平凡算法
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

//cache优化算法
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

//验证结果是否相等
bool compareresults(const vector<int>& result1, const vector<int>& result2, int n) {
    const int epsilon = 1e-9;
    for (int i = 0; i < n; i++) {
        if (abs(result1[i] - result2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

//计时
class counttime {
private:
    LARGE_INTEGER m_frequency;
    LARGE_INTEGER m_start;
    LARGE_INTEGER m_end;
public:
    counttime() {
        QueryPerformanceFrequency(&m_frequency);
    }
    void start() {
        QueryPerformanceCounter(&m_start);
    }
    void stop() {
        QueryPerformanceCounter(&m_end);
    }
    double getelapsedmilliseconds() const {
        return static_cast<double>(m_end.QuadPart - m_start.QuadPart) * 1000.0 / m_frequency.QuadPart;
    }
};

//测试性能
struct testresult {
    int size;
    double ordinarytime;      // 平凡算法耗时(ms)
    double cachetime;  // 优化算法耗时(ms)
    double speedup;        // 加速比
    double matrixsizemb;   // 矩阵大小(MB)
    bool correct;          // 结果是否正确
};

//测试
testresult test(int n) {
    testresult result;
    result.size = n;
    result.matrixsizemb = (n * n * sizeof(int)) / (1024* 1024);

    // 生成测试数据
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
    result.ordinarytime= totaltime1 / repeatcount;

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

    // 验证结果
    result.correct = compareresults(result1, result2, n);
    result.speedup = result.ordinarytime / result.cachetime;
    return result;
}

//输出结果
void printresult(const vector<testresult>& results) {
    cout << "\\n" << string(120, '=') << endl;
    cout << left << setw(10) << "规模"
        << setw(12) << "矩阵大小(MB)"
        << setw(18) << "平凡算法(ms)"
        << setw(18) << "优化算法(ms)"
        << setw(12) << "加速比"
        << "状态" << endl;
    cout << string(120, '-') << endl;

    cout << fixed << setprecision(3);
    for (const auto& r : results) {
        cout << left << setw(10) << r.size
            << setw(12) << r.matrixsizemb
            << setw(18) << r.ordinarytime
            << setw(18) << r.cachetime
            << setw(12) << r.speedup
            << (r.correct ? "true" : "false") << endl;
    }
    cout << string(120, '=') << endl;
}

//导入数据到CSV文件
void ExportToCSV(const vector<testresult>& results, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法创建文件: " << filename << endl;
        return;
    }

    file << "matrixsize,matrixsizemb,ordinarytime_ms,cachetime_ms,speedup\\n";

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

    // 生成测试规模（约20-30个数据点）
    vector<int> testSizes = generatetestmatrix();
    cout << "\\n生成 " << testSizes.size() << " 个测试规模" << endl;
    cout << "范围: " << testSizes.front() << " ~ " << testSizes.back() << endl;

    // 运行所有测试
    vector<testresult> results;
    int current = 0;

    cout << "\\n开始性能测试..." << endl;
    for (int n : testSizes) {
        current++;
        cout << "进度: [" << current << "/" << testSizes.size() << "] 测试规模 " << n << " ... " << flush;

        testresult r =test(n);
        results.push_back(r);

        cout << "完成 (平凡算法: " << fixed << setprecision(3) << r.ordinarytime
            << " ms, 优化算法: " << r.cachetime << " ms, 加速比: "
            << setprecision(2) << r.speedup << "x)" << endl;
    }

    // 输出结果
    printresult(results);

    // 导出CSV文件
    ExportToCSV(results, "performance_data.csv");

    cout << "\\n按任意键退出..." << endl;
    cin.get();
    return 0;
}