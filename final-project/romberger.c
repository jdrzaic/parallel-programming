#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>

#define block_first(i,T,N)		(((i) * (N)) / (T))

double func(double value) {
	printf("value: %lf\n", value);
	return value;
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
	int last = block_first(rank + 1, size, twotop + 1);
	printf("last: %d\n", last);
	double* trap = (double*)calloc(p + 1, sizeof(double));
	double* recv = (double*)malloc((p + 1) * sizeof(double));
	for(int i = first; i < last; ++i) {
		double point = i * h;
		double f_value = func(point);
		int to = first_one(i);
		if(i == 0) {
			to = p - 1;
			f_value /= 2;
		}
		if(i == twotop) f_value /= 2;
		for(int j = 0; j <= to; ++j) {
			trap[j] += f_value;
		}
	}

	printf("%d", rank);
	for(int i = 0; i < p + 1; ++i) {
		printf("%lf\n", trap[i]);
	}

	MPI_Reduce(trap, recv, p + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0) {
		for(int i = 0; i < p + 1; ++i) {
			printf("apr: %lf\n", recv[i]);
		}
 	}
	MPI_Finalize();
	return 0;
}

