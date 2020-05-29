#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdlib>
#include <iostream>

// ����������� ������� �������� 
#define MAX_ITER 100000
// ������������ ������� �������
#define MAX 100
// �������
#define TOL 0.000001

// ��������� ��������� ������� �������� �� ���� ��������
float rand_float(int max) {
    return ((float)rand() / (float)(RAND_MAX)) * max;
}


// ��������� �������
void allocate_init_2Dmatrix(float*** mat, int n, int m) {
    int i, j;
    *mat = (float**)malloc(n * sizeof(float*));
    for (i = 0; i < n; i++) {
        (*mat)[i] = (float*)malloc(m * sizeof(float));
        for (j = 0; j < m; j++)
            (*mat)[i][j] = rand_float(MAX);
    }
}

// ��������
void solver(float*** mat, int n, int m) {
    float diff = 0, temp;
    bool done = false;
    int cnt_iter = 0, i, j;

    while (!done && (cnt_iter < MAX_ITER)) {
        // ����� ��� �������� ������� 
        diff = 0;
        // �������� ���� ������� ��� ������ � ������� ����� �� ��������, ���� �� ����������
        for (i = 1; i < n - 1; i++)
            for (j = 1; j < m - 1; j++) {
                temp = (*mat)[i][j];
                (*mat)[i][j] = 0.2 * ((*mat)[i][j] + (*mat)[i][j - 1] + (*mat)[i - 1][j] + (*mat)[i][j + 1] + (*mat)[i + 1][j]);
                diff += std::abs((*mat)[i][j] - temp);
            }
        // ���� ������� ���������, �� �������� ��������
        if (diff / n / n < TOL)
            done = true;
        cnt_iter++;
    }

    if (done)
        std::cout << "Solver converged after " << cnt_iter << " iterations\n";
    else
        std::cout << "Solver not converged after " << cnt_iter << " iterations\n";

}

int main() {
    int n;
    float** a;

    //��������� �������
    std::cin >> n;

    //��������� �������
    allocate_init_2Dmatrix(&a, n, n);

    // ������ ���
    clock_t i_exec_t = clock();

    //��� ��������
    solver(&a, n, n);

    // ʳ���� ������ ��������� - ������� ���
    clock_t f_exec_t = clock();
    float exec_time = (float)(f_exec_t - i_exec_t) / CLOCKS_PER_SEC;
    std::cout << "Operations time: " <<  exec_time << std::endl;

    return 0;
}