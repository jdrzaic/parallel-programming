
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>

unsigned long long int block_first(int i, int T, unsigned long long int N) {
	double re = (double)i / T;
	return (unsigned long long int)(re * N);
}

double func(double value) {
//	return pow(M_E, value);
//	return sin(17 * value * M_PI);
//	return cos(2 * value * M_PI);
//	return pow(M_E, 13 * value);
//	return cos(7 * value);
//	return value * value * value;
	return pow(value, 1.5);
}

double calc_value(double n2, double n, int k) {
	double v = pow(4, k);
	return (v * n) / (v - 1) - n2 / (v - 1);
}

int max(int a, int b) {
	return a < b ? b : a;
}

int first_one(int n) {
	int pos = 0;
	if(n == 0) return 0;
	while((n >> pos) % 2 == 0) pos++;
	return pos;
}

int next_proc(int rank, int size, int n) {
	for(int i = rank + 1; i < size; ++i) {
		if(block_first(i, size, n) != block_first(i + 1, size, n)) {
			return i;
		}
	}
	return -1;
}

int prev_proc(int rank, int size, int n) {
	for(int i = rank - 1; i >= 0; --i) {
		if(block_first(i, size, n) != block_first(i + 1, size, n)) {
			return i;
		}
	}
	return -1;
}

int last_proc(int size, int n) {
	for(int i = size - 1; i >= 0; --i) {
		if(block_first(i, size, n) != block_first(i + 1, size, n)) {
			return i;
		}
	}
	return -1;
}

int first_proc(int size, int n) {
	for(int i = 0; i < size; ++i) {
		if(block_first(i, size, n) != block_first(i + 1, size, n)) {
			return i;
		}
	}
	return -1;
}

int main(int argc, char** argv) {
	int size;
	int rank;

	double a, b, h;
	int p;
	MPI_Status status;
	MPI_Request request;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	a = atof(argv[1]);
	b = atof(argv[2]);
	p = atoi(argv[3]);
	unsigned long long int twotop = 1LL << p;
	h = (b - a) / twotop;
	//index of first point for this process
	unsigned long long int first = block_first(rank, size, twotop + 1);
	//after last
	unsigned long long int last = block_first(rank + 1, size, twotop + 1);
	printf("rank: %d, first point: %llu, last point + 1: %llu, size: %d, points: %llu\n", rank, first, last, size, twotop);
	double* trap = (double*)calloc(p + 1, sizeof(double));
	double* recv = (double*)malloc((p + 1) * sizeof(double));
	unsigned long long int i = first;
	
	while(i < last) {
		double point = a + i * h;
		double f_value = func(point);
		int to = first_one(i);
		if(i == 0) {
			to = p;
			f_value /= 2;
		}
		if(i == twotop) f_value /= 2;
		double htmp = h;
		for(int j = 0; j <= to; ++j) {
			trap[j] += (f_value * htmp);
			htmp *= 2;
		}
		++i;
	}

	MPI_Allreduce(trap, recv, p + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if(rank == 0) {
		for(int i = 0; i < p+1; ++i) {
			printf("trap: %.*e\n", 12, recv[i]);
		}
	}

	int first_row = block_first(rank, size, p + 1);
	int last_row = block_first(rank + 1, size, p + 1);
	printf("rank: %d, first_row: %d, last_row+1: %d, \n", rank, first_row, last_row);
	if(first_row == last_row) {
		goto end;
	}
	double* table = (double*)calloc(last_row - first_row, sizeof(double));
	for(int i = 0; i < last_row - first_row; ++i) {
		table[i] = recv[p - first_row - i];
	}
	
	for(int j = last_row - 1; j >= max(1, first_row); --j) {
		table[j - first_row] = calc_value(recv[p - j + 1], recv[p - j], 1);
	}

	for(int i = 2; i <= last_row; ++i){
		double got;
		if(rank < last_proc(size, p + 1) && rank % 2) {
			//printf("process %d sends %lf to process %d in column %d\n", rank, table[last_row - first_row - 1], next_proc(rank, size, p + 1), i);
			MPI_Send(&table[last_row - first_row - 1], 1, MPI_DOUBLE, next_proc(rank, size, p + 1), 0, MPI_COMM_WORLD);
			//printf("process %d sent %lf to process %d in column %d\n", rank, table[last_row - first_row - 1], next_proc(rank, size, p + 1), i);
                } else if(rank > first_proc(size, p + 1) && rank % 2 == 0 && i <= first_row) {
				//printf("%d waits for process %d to send in column %d\n", rank, rank - 1, i);
				MPI_Recv(&got, 1, MPI_DOUBLE,  prev_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("process %d recieved %lf from process %d in column %d\n", rank, got, rank - 1, i);
		}
		if(rank < last_proc(size, p + 1) && rank % 2 == 0) {
			//printf("process %d sends %lf to process %d in column %d\n", rank, table[last_row - first_row - 1], next_proc(rank, size, p + 1), i);
			MPI_Send(&table[last_row - first_row - 1], 1, MPI_DOUBLE,  next_proc(rank, size, p + 1), 0, MPI_COMM_WORLD);
			//printf("process %d sent %lf to process %d in column %d\n", rank, table[last_row - first_row - 1], next_proc(rank, size, p + 1), i);	
		} else if(rank > first_proc(size, p + 1) && rank  % 2 && i <= first_row) {
			//printf("%d waits for process %d to send in column %d\n", rank, rank - 1, i);
			MPI_Recv(&got, 1, MPI_DOUBLE,  prev_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("process %d recieved %lf from process %d in column %d\n", rank, got, rank - 1, i);
		}
		if(i < last_row) {
			for(int j = last_row - 1; j > max(i, first_row); --j) {
				table[j - first_row] = calc_value(table[j - first_row - 1], table[j - first_row], i);
//				printf("col: %d, row: %d value: %lf\n", i, j, table[j - first_row]);
			}
			int l = max(i, first_row);
			if(rank > first_proc(size, p + 1) && i <= first_row) {
				table[l - first_row] = calc_value(got, table[l - first_row], i);
			} else {
				table[l - first_row] = calc_value(table[l - first_row - 1], table[l - first_row], i);
			}
			printf("col: %d, row: %d value: %lf\n", i, l, table[l - first_row]);
		}
	}
	if(rank == last_proc(size, p + 1)) {
		printf("aproksimacija: %.16lf\n", table[last_row - first_row - 1]);
	}
end:	free(recv);
	free(table);
	free(trap);
	MPI_Finalize();
	return 0;
}

