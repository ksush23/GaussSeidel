#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// ����������� ������� �������� 
#define MAX_ITER 100000
// ������������ ������� �������
#define MAX 100
// �������
#define TOL 0.000001


// ��������� ��������� ������� �������� �� ���� ��������
float rand_float(const int max) {
	return ((float)rand() / (float)(RAND_MAX)) * max;
}




// ϳ�������� ���� ������� ����� ��� ������� ������
int get_max_rows(const int num_threads, const int n) {
	return (int)(ceil((n - 2) / num_threads) + 2);
}




// ���������� ������� ���������� ����������
void alloc_matrix(float** mat, const int n, const int m) {

	*mat = (float*)malloc(n * m * sizeof(float));

	const int size = n * m;
	for (int i = 0; i < size; i++) {
		(*mat)[i] = rand_float(MAX);
	}
}





// ����������� ��������
void solver(float** mat, const int n, const int m, const int num_ths, const int max_cells_per_th) {

	float diff;

	bool done = false;
	int cnt_iter = 0;
	const int mat_dim = n * n;

	while (!done && (cnt_iter < MAX_ITER)) {

		// ����� ��� �������� ������� 
		diff = 0;
		// ��������� ���������� ������ �� ��������� ��������� parallel 
		// ��� ����, ��� ������ ���� ��������� ��� ��������, � �� �� ��������, ������������� ��������� for
		// ���� ��������� ������������ ��������� ����� ����� ���������� �������, �� ���� ����������� ����� �� ������ ����������
		// ���� ������������� collapse, ��� ������� ������� ������������� �����, �� ������ ������������ � ����������
		// schedule ���� ���� ����� �������� �������������� �� ��������, static ������� ������-�������� �������
		// reduction ���� �������� � ������ ������, ��� ����� �� ���� ������������
		#pragma omp parallel for num_threads(num_ths) schedule(static, max_cells_per_th) collapse(2) reduction(+:diff)
		// �������� ���� ������� ��� ������ � ������� ����� �� ��������, ���� �� ����������
		for (int i = 1; i < n - 1; i++) {
			for (int j = 1; j < m - 1; j++) {

				const int pos = (i * m) + j;
				const float temp = (*mat)[pos];

				(*mat)[pos] =
					0.2f * (
						(*mat)[pos]
						+ (*mat)[pos - 1]
						+ (*mat)[pos - n]
						+ (*mat)[pos + 1]
						+ (*mat)[pos + n]
						);

				diff += abs((*mat)[pos] - temp);
			}
		}
		// ���� ������� ���������, �� �������� ��������
		if (diff / mat_dim < TOL) {
			done = true;
		}
		cnt_iter++;
	}

	if (done)
		std::cout << "Solver converged after " << cnt_iter << " iterations\n";
	else
		std::cout << "Solver not converged after " << cnt_iter << " iterations\n";
}




int main() {

	int n;
	std::cin >> n;

	// ������� ���
	const double i_total_t = omp_get_wtime();

	float* mat;
	alloc_matrix(&mat, n, n);

	// ���������� ������� ������ ��� ������� ������
	const int max_threads = omp_get_max_threads();
	const int max_rows = get_max_rows(max_threads, n);
	const int max_cells = max_rows * (n - 2);

	// ������� ��� ��������
	const double i_exec_t = omp_get_wtime();

	// ����������� ��������
	solver(&mat, n, n, max_threads, max_cells);

	// ��������� ��������� ���� ���������
	const double f_exec_t = omp_get_wtime();


	free(mat);

	// ��������� ��������� ����
	const double f_total_t = omp_get_wtime();

	const double total_time = f_total_t - i_total_t;
	const double exec_time = f_exec_t - i_exec_t;
	std::cout << "Total time: " << total_time << "\n";
	std::cout << "Operations time: " << exec_time << "\n";
}
