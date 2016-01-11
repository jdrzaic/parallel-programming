
#include<stdio.h>
#include<complex.h>
#include<mpi.h>
#include<stdlib.h>
#include<math.h>

#define block_first(i,T,N)      (((i) * (N)) / (T))
#define block_last(i,T,N)       (block_first(i+1,T,N) - 1)
#define block_size(i,T,N)       (block_first(i+1,T,N) - block_first(i,T,N))

double complex* read_matrix(char* file, int n) {
        FILE* m_file = fopen(file, "rb");
    	if(!m_file) {
        	return NULL;
    	}
    	double complex *a;
   	if((a = (double complex*)malloc(n * sizeof(double complex))) == NULL) {
       		fprintf(stderr, "error alocating memory.\n");
        	fclose(m_file);
    		return NULL;
    	}
    	if(fread(a, sizeof(double complex), n, m_file) != n) {
        	fclose(m_file);
        	return NULL;
    	}
    	fclose(m_file);
    	return a;
}

int main(int argc, char** argv) {
	int size;
	int rank;

        double complex* v;
	double complex max;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n;
        if(argc == 3) {
                n = atoi(argv[1]);
	}else {
		MPI_Finalize();
		return 1;
	}
	if(size < 1) {
		if(rank == size - 1) {
			printf("Number of processes must be positive!\n");
		}
		MPI_Finalize();
		return 1;
	}
	if(argc != 3 || n < 1) {
		if(rank == size - 1) {
			printf("error using command line arguments!\n");
		}
		MPI_Finalize();
		return 1;
	}
	int elnum = block_size(rank, size, n);

        if(rank == size - 1) {
		FILE* file = fopen(argv[2], "rb");
		if(!file) {
			printf("Unable to open file %s\n!", argv[2]);
		}
		v = read_matrix(argv[2], n);
		if(v == NULL) {
			printf("error reading from file!\n");
		}
                for(int i = 0; i < size - 1; ++i) {
                        int offset = block_first(i, size, n);
                        int selnum = block_size(i, size, n);
                        MPI_Send(v + offset, selnum, MPI_C_DOUBLE_COMPLEX, i, 0, MPI_COMM_WORLD);
                }
		for(int i = 0; i < n; ++i) {
			printf("%lf, %lf\n", creal(v[i]), cimag(v[i]));
		}
		v = v + block_first(rank, size, n);
	} else {
                v = (double complex*)malloc(elnum * sizeof(double complex));
		MPI_Recv(v, elnum, MPI_C_DOUBLE_COMPLEX, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("process %d recieved %d elements\n", size - 1 - rank, elnum);
        }
	double max_norm;
	for(int i = 0; i < elnum; ++i) {
		double norm = cabs(v[i]);
		if(i == 0 || norm > max_norm) {
			max_norm = norm;
		}
	}
	int idx = size - 1 - rank;
	int shift;
	for(shift = 1; shift < size; shift <<= 1) {
		if((idx == 0 || (idx & -idx) > shift) && (idx + shift < size)) {
			double snorm;
			int from = rank - shift;
			MPI_Recv(&snorm, 1, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(snorm > max_norm) max_norm = snorm;
		}
		else{
			int sidx = idx - shift;
			if((sidx == 0 || (sidx & -sidx) > shift) && (sidx >= 0)) {
				int send_to = rank + shift;
				MPI_Send(&max_norm, 1, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);
			}
		}
	}
	shift >>= 1;

	for(; shift > 0; shift >>= 1) {
		if((idx == 0 || (idx & -idx) > shift) && (idx + shift < size)) {
			int send_to = rank - shift;
			MPI_Send(&max_norm, 1, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);
                }
                else {
			int sidx = idx - shift;
			if((sidx == 0 || (sidx & -sidx) > shift) && (sidx >= 0)) {
				int rec_from = rank + shift;
				MPI_Recv(&max_norm, 1, MPI_DOUBLE, rec_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
                }

	}
	if(max_norm == 0.0) {
		if(rank = size - 1) {
			printf("norma: 0.0");
		}
		MPI_Finalize();
		return 0;
	}
	double eucl = 0.0;
        for(int i = 0; i < elnum; ++i) {
                double abs = cabs(v[i]);
                eucl += (abs * abs) / (max_norm * max_norm);
        }

	for(shift = 1; shift < size; shift <<= 1) {
                if((idx == 0 || (idx & -idx) > shift) && (idx + shift < size)) {
                        double seucl;
                        int from = rank - shift;
                        MPI_Recv(&seucl, 1, MPI_DOUBLE, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        eucl += seucl;
                }
                else{
                        int sidx = idx - shift;
                        if((sidx == 0 || (sidx & -sidx) > shift) && (sidx >= 0)) {
                                int send_to = rank + shift;
                                MPI_Send(&eucl, 1, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);
                        }
                }
        }


	if(idx == 0) {
		eucl = sqrt(eucl) * max_norm;
		printf("norma=%3.14f", eucl);
	}

	MPI_Finalize();
	return 0;
}
