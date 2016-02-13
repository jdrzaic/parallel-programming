#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>

#define block_first(i,T,N)		(((i) * (N)) / (T))

double func(double value) {
	return pow(M_E, value);
}

double calc_value(double n2, double n, int k) {
	double v = pow(4, k);

	return (v * n - n2) / (v - 1);
}

int max(int a, int b) {
	return a < b ? b : a;
}

double func2(double value) {
	return value * value;
}

int first_one(int n) {
	int pos = 0;
	if(n == 0) return 0;
	while((n >> pos) % 2 == 0) pos++;
	return pos;
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
	int twotop = 1 << p;
	h = (b - a) / twotop;
	//index of first point for this process
	int first = block_first(rank, size, twotop + 1);
	printf("first: %d\n", first);
	//after last
	int last = block_first(rank + 1, size, twotop + 1);
	printf("last: %d\n", last);
	double* trap = (double*)calloc(p + 1, sizeof(double));
	double* recv = (double*)malloc((p + 1) * sizeof(double));
	for(int i = first; i < last; ++i) {
		double point = a + i * h;
		double f_value = func(point);
//		printf("f:  %lf\n", f_value);
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
	}

	MPI_Allreduce(trap, recv, p + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for(int i = 0; i < p+1; ++i) {
		printf("i: %.*e\n", 12, recv[i]);
	}

	int first_row = block_first(rank, size, p + 1);
	int last_row = block_first(rank + 1, size, p + 1);
	printf("first: %d last %d\n", first_row, last_row);
	double* table = (double*)calloc(last_row - first_row, sizeof(double));
	for(int i = 0; i < last_row - first_row; ++i) {
		table[i] = recv[p - first_row - i];
		//printf("%d %lf\n", i, table[i]);
	}
	for(int i = 1; i < last_row; ++i){
		if(rank < size - 1) {
                        MPI_Send(&table[last_row - first_row - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                        printf("process %d sent %lf to process %d in column %d\n", rank, table[last_row - first_row - 1], rank + 1, i);
                }

		for(int j = last_row - 1; j > max(i, first_row); --j) {
			table[j - first_row] = calc_value(table[j - first_row - 1], table[j - first_row], i);
		}
		//if(rank < size - 1) {
		//	MPI_Isend(&table[last_row - first_row - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
		//	printf("process %d sent %lf to process %d in column %d:", rank, table[last_row - first_row - 1], rank + 1, i);
                //}
		int l = max(i, first_row);
		if(rank > 0) {
			double got = 4;
			MPI_Recv(&got, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("%d %lf recieved", rank, got);
			table[l - first_row] = calc_value(got, table[l - first_row], i);
		} else {
			table[l] = calc_value(table[l - 1], table[l], i);
		}
		printf("rank: %d first: %d\n", rank, l);
		for(int t = l - first_row; t < last_row - first_row; ++t) {
			printf("%d index: %d value: %lf\n", rank, t, table[t]);
		}
	}
	if(rank == size - 1) {
		printf("aproksimacija: %.*e", 14, table[last_row - first_row - 1]);
	}
	MPI_Finalize();
	return 0;
}

