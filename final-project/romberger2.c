
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>
#include<string.h>

unsigned long long int block_first(int i, int T, unsigned long long int N) {
	double re = (double)i / T;
	return (unsigned long long int)(re * N);
}

double func(double value) {
//  return pow(M_E, value);
	return sin(17 * value * M_PI);
//	return cos(2 * value * M_PI);
//	return pow(M_E, 13 * value);
//	return cos(7 * value);
//	return value * value * value;
//	return pow(value, 1.5);
//  return pow(value, 0.5);
//  return value;
}

double calc_value(double n2, double n, double v) {
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
    int verb = 0;

	double a, b, h;
	int p;
    char v[8] = "verbose";
	MPI_Status status;
	MPI_Request request_recv, request_send;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(argc != 4) {
        if(argc == 5 && strcmp(v, argv[4]) == 0) {
            verb = 1;
        }
        else {
            goto end2;
        }
    }
	a = atof(argv[1]);
	b = atof(argv[2]);
	p = atoi(argv[3]);
	unsigned long long int twotop = 1LL << p;
	h = (b - a) / twotop;
	unsigned long long int first = block_first(rank, size, twotop + 1);
	unsigned long long int last = block_first(rank + 1, size, twotop + 1);
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
	int first_row = block_first(rank, size, p + 1);
	int last_row = block_first(rank + 1, size, p + 1);
    double* all;
    int elems = first_row * (last_row - first_row) + (last_row - first_row) * (last_row - first_row + 1) / 2;
    if(first_row == last_row) {
        goto end;
    }
    if (verb) {
        all = (double*)calloc(elems, sizeof(double));
        for (int j = first_row; j < last_row; ++j) {
            all[first_row * (j - first_row) + (j + 1 - first_row) * (j - first_row) / 2] = recv[p - j];
        }
    }
	
	double* table = (double*)calloc(last_row - first_row, sizeof(double));
	double fourtok = 4;
	for(int j = last_row - 1; j >= max(1, first_row); --j) {
		table[j - first_row] = calc_value(recv[p - j + 1], recv[p - j], fourtok);
	}
    
    for(int j = max(1, first_row); j < last_row; ++j) {
        all[first_row * (j - first_row) + (j - first_row + 1) * (j - first_row) / 2 + 1] = table[j - first_row];
    }

	for(int i = 2; i <= last_row; ++i){
		fourtok *= 4;
		double got;
        /*if (rank < last_proc(size, p + 1)) {
            MPI_Isend(&table[last_row - first_row - 1], 1, MPI_DOUBLE, next_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, &request_send);
        }
        else if (rank > first_proc(size, p + 1) && i <= first_row) {
            MPI_Irecv(&got, 1, MPI_DOUBLE,  prev_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, &request_recv);
        }
        if(rank < last_proc(size, p + 1)) MPI_Wait(&request_send, &status);
        else if(rank > first_proc(size, p + 1) && i <= first_row) MPI_Wait(&request_recv, &status);
        */
		if(rank < last_proc(size, p + 1) && rank % 2) {
			MPI_Send(&table[last_row - first_row - 1], 1, MPI_DOUBLE, next_proc(rank, size, p + 1), 0, MPI_COMM_WORLD);
		} else if(rank > first_proc(size, p + 1) && rank % 2 == 0 && i <= first_row) {
			MPI_Recv(&got, 1, MPI_DOUBLE,  prev_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if(rank < last_proc(size, p + 1) && rank % 2 == 0) {
			MPI_Send(&table[last_row - first_row - 1], 1, MPI_DOUBLE,  next_proc(rank, size, p + 1), 0, MPI_COMM_WORLD);
		} else if(rank > first_proc(size, p + 1) && rank  % 2 && i <= first_row) {
			MPI_Recv(&got, 1, MPI_DOUBLE,  prev_proc(rank, size, p + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
        
		if(i < last_row) {
			for(int j = last_row - 1; j > max(i, first_row); --j) {
				table[j - first_row] = calc_value(table[j - first_row - 1], table[j - first_row], fourtok);
                all[first_row * (j - first_row) + (j + 1 - first_row) * (j - first_row) / 2 + i] = table[j - first_row];
			}
			int l = max(i, first_row);
			if(rank > first_proc(size, p + 1) && i <= first_row) {
				table[l - first_row] = calc_value(got, table[l - first_row], fourtok);
			} else {
				table[l - first_row] = calc_value(table[l - first_row - 1], table[l - first_row], fourtok);
                
			}
            all[first_row * (l - first_row) + (l + 1 - first_row) * (l - first_row) / 2 + i] = table[l - first_row];
		}
	}
    if(verb) {
        if(rank > first_proc(size, p + 1)) {
            MPI_Send(all, elems, MPI_DOUBLE, first_proc(size, p + 1), 0, MPI_COMM_WORLD);
        } else {
            double* allp = (double*) calloc((p + 2) * (p + 1) / 2, sizeof(double));
            for(int i = 0; i < elems; ++i) {
                allp[i] = all[i];
            }
            if (size > 1) {
                int i = first_proc(size, p + 1);
                while (1) {
                    i = next_proc(i, size, p + 1);
                    int first = block_first(i, size, p + 1);
                    int offset = (first + 1) * first / 2;
                    int last_rows = block_first(i + 1, size, p + 1);
                    int first_rows = block_first(i, size, p + 1);
                    int recv = first_rows * (last_rows - first_rows) + (last_rows - first_rows) * (last_rows - first_rows + 1) / 2;
                    MPI_Recv(allp + offset, recv, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if(i == last_proc(size, p + 1)) break;
                }
            }
            int k = 0;
            for(int i = 0; i < p + 1; ++i) {
                for(int j = 0; j <= i; ++j) {
                    printf("% .8lf  ", allp[k++]);
                }
                printf("\n");
            }
            free(allp);
        }
    }
    if(rank == last_proc(size, p + 1)) {
        printf("aproksimacija: %.16lf\n", table[last_row - first_row - 1]);
	}
end:	free(recv);
	free(table);
	free(trap);
    free(all);
end2:	MPI_Finalize();
	return 0;
}

