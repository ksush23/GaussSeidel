#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>
#include "mpi.h"
#include <iostream>

// Максимальна кількість ітерацій 
#define MAX_ITER 100000
// Максимальний елемент матриці
#define MAX 100
// Точність
#define TOL 0.000001


// Генерація рандомних значень відходячи від макс значення
float rand_float(int max) {
	return ((float)rand() / (float)(RAND_MAX)) * max;
}




// Підрахунок макс кількості рядків для кожного процесу
int get_max_rows(int num_nodes, int n) {
	return (int)(ceil((n - 2) / num_nodes) + 2);
}




// Отримуємо позицію, з якої елементи будуть відсилатись чи отримуватись
int get_node_offset(int node_id, int n, int max_rows) {
	return node_id * n * (max_rows - 2);
}




// Підрахунок кількості елементів для кожного процесу
int get_node_elems(int node_id, int n, int max_rows) {

	int node_offset = get_node_offset(node_id, n, max_rows);
	int node_elems = max_rows * n;

	// Загальний випадок
	if (node_offset + node_elems <= (n * n)) {
		return node_elems;
	}

	// Випадок кінця, де може бути менше елементів
	else {
		return (n * n) - node_offset;
	}
}




// Створення матриці
void allocate_root_matrix(float** mat, int n, int m) {

	*mat = (float*)malloc(n * m * sizeof(float));
	for (int i = 0; i < (n * m); i++) {
		(*mat)[i] = rand_float(MAX);
	}
}




// Матриця для процесів
void allocate_node_matrix(float** mat, int num_elems) {
	*mat = (float*)malloc(num_elems * sizeof(float));
}




// Паралельний алгоритм розв язку для заданої кількості елементів
void solver(float** mat, int n, int num_elems) {

	float diff = 0, temp;
	bool done = false;
	int cnt_iter = 0, myrank;

	while (!done && (cnt_iter < MAX_ITER)) {
		diff = 0;

		// Пропускаємо перший та останній рядки
		for (int i = n; i < num_elems - (2 * n); i++) {

			// І стовпці
			if ((i % n == 0) || (i + 1 % n == 0)) {
				continue;
			}

			int pos_up = i - n;
			int pos_do = i + n;
			int pos_le = i - 1;
			int pos_ri = i + 1;

			temp = (*mat)[i];
			(*mat)[i] = 0.2 * ((*mat)[i] + (*mat)[pos_le] + (*mat)[pos_up] + (*mat)[pos_ri] + (*mat)[pos_do]);
			diff += abs((*mat)[i] - temp);
		}

		// Якщо точність досягнута, то закінчуємо алгоритм
		if (diff / n / n < TOL) {
			done = true;
		}
		cnt_iter++;
	}


	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (done) {
		std::cout << "Node " << myrank << ": Solver converged after " << cnt_iter << " iterations\n";
	}
	else {
		std::cout << "Node " << myrank << ": Solver not converged after " << cnt_iter << " iterations\n";
	}
}




int main(int argc, char* argv[]) {

	int np, myrank, n, communication, i;
	float* a = 0, * b = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	n = atoi(argv[1]);
	// Головна відміннсть колективних операцій від операцій точка-точка - це той факт, що в колективних операціях беручь участь всі процеси комунікатори
	communication = atoi(argv[2]);



	// Обчислюємо кількість значень для кожного процесу
	int max_rows = get_max_rows(np, n);

	// Масиви для зберігання "відправної точки" та кількості елементів для кожного процесу
	int nodes_offsets[MAX];
	int nodes_elems[MAX];
	for (i = 0; i < np; i++) {
		nodes_offsets[i] = get_node_offset(i, n, max_rows);
		nodes_elems[i] = get_node_elems(i, n, max_rows);
	}

	// Локальна змінна зберігання поточної кількості елементів
	int num_elems = nodes_elems[myrank];
	// Фіксуємо час
	double tscom1 = MPI_Wtime();

	switch (communication) {
	case 0: {
		if (myrank == 0) {

			// Вся матриця
			allocate_root_matrix(&a, n, n);

			// Майстер-процус відправляє повідомлення всім іншим
			for (i = 1; i < np; i++) {
				int i_offset = nodes_offsets[i];
				int i_elems = nodes_elems[i];
				// Вказуємо буфер передачі повідомлення(адрес початку, кількість та тип) і атрибути повідомлення(номер процесу отримувача, ідентифікатор
				// повідомлення та комунікатор)
				MPI_Send(&a[i_offset], i_elems, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
		}
		else {

			// Матриця для отриманих рядків
			allocate_node_matrix(&a, num_elems);
			MPI_Status status;

			// Отримуємо повідомлення
			// Вказуємо буфер отримки повідомлення(адрес початку, кількість, тип) і атрибути вибору приймаючого повідомлення(номер процеса-відправника, 
			// ідентифікатор повідомлення, комунікатор) та статус
			MPI_Recv(a, num_elems, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
		break;
	}
	
	case 1: {
		if (myrank == 0) {
			// Вся матриця
			allocate_root_matrix(&a, n, n);
		}
		// Конкретна ділянка пам яті, де обчислюються отримані рядки
		allocate_node_matrix(&b, num_elems);
		// Вказуємо буфер прийому повідомлення, далі буфер передачі повідомлення, root і комунікатор
		MPI_Scatterv(a, nodes_elems, nodes_offsets, MPI_FLOAT, b, num_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);
		break;
	}
	}


	double tfcom1 = MPI_Wtime();
	double tsop = MPI_Wtime();

	if (communication == 0) {
		solver(&a, n, num_elems);
	}
	else {
		solver(&b, n, num_elems);
	}

	double tfop = MPI_Wtime();
	double tscom2 = MPI_Wtime();


	switch (communication) {
	case 0: {
		//Майстер-потік "збирає" всі повідомлення назад
		if (myrank == 0) {

			MPI_Status status;

			for (i = 1; i < np; i++) {
				int i_offset = nodes_offsets[i] + n;
				int i_elems = nodes_elems[i] - (2 * n);
				MPI_Recv(&a[i_offset], i_elems, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}
		}
		else {
			int solved_offset = n;
			int solved_elems = num_elems - (2 * n);
			MPI_Send(&a[solved_offset], solved_elems, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}

		break;
	}

	case 1: {

		// Вказуємо буфер передачі повідомлення(кожний процес передає дані зі свого буферу на процес із заданим номером root), далі буфер прийому повідомлення
		// (використовується тільки на процесі-отримувачі, який і розміщує дані там), root і комунікатор
		MPI_Gatherv(b, num_elems, MPI_FLOAT, a, nodes_elems, nodes_offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
		break;
	}
	}

	double tfcom2 = MPI_Wtime();


	if (myrank == 0) {
		float com_time = (tfcom1 - tscom1) + (tfcom2 - tscom2);
		float ops_time = tfop - tsop;
		float total_time = com_time + ops_time;

		std::cout << "Communication time: " << com_time << "\n";
		std::cout << "Operations time: " << ops_time << "\n";
		std::cout << "Total time: " << total_time << "\n";
	}


	free(a);
	free(b);

	MPI_Finalize();
	return 0;
}