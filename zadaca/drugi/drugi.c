#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<time.h>
#include<sys/time.h>

#define block_first(i,T,N)		(((i) * (N)) / (T))
#define block_last(i,T,N)		(block_first(i+1,T,N) - 1)
#define block_size(i,T,N)		(block_first(i+1,T,N) - block_first(i,T,N))

extern void dgemv_(char* T, int* M, int* N, double* ALPHA, double* A,
    int *LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);

extern double ddot_(int* N, double* DX, int* INCX, double* DY, int* INCY);

typedef struct {
    int start_row;
    int end_row;
    int start_col;
    int end_col;
    int n;
    int m;
    int N;
    double w;
    double* a;
    double* x;
    double* r;
    double* d;
    double* norms;
    pthread_barrier_t* bar_ptr;
} thr_struct;

void calc_r(thr_struct* str) {
	int size = str->end_row - str->start_row + 1, inc = 1;
	double alpha = -1.0, beta = 1.0;
	char tr = 'T';
	dgemv_(&tr, &(str->m), &size, &alpha, str->a  + str->start_row * str->m,
	    &(str->m), str->d, &inc, &beta, str->r + str->start_row, &inc);
}

void calc_norms(thr_struct* str) {
	for(int i = str->start_col; i <= str->end_col; ++i) {
		double norm = ddot_(&(str->n), str->a + i, &(str->m), str->a + i, &(str->m));
		str->norms[i] = norm;
 	}
}

void update_x(thr_struct* str) {
	int inc = 1;
	for(int i = str->start_col; i <= str->end_col; ++i) {
		double dot = ddot_(&(str->n), str->r, &inc, str->a + i, &(str->m));
		str->d[i] = str->w * dot / str->norms[i];
		str->x[i] += str->d[i];
	}
}

void* thr_func(void* arg) {
	thr_struct* str = (thr_struct*)arg;
	calc_r(str);
	calc_norms(str);
	for(int i = 0; i < str->N; ++i) {
		pthread_barrier_wait(str->bar_ptr);
		update_x(str);
		pthread_barrier_wait(str->bar_ptr);
		calc_r(str);
	}
}

double* read_matrix(char* file, int m, int n) {
    FILE* m_file = fopen(file, "rb");
    if(!m_file) {
        return NULL;
    }
    double *a;
    if((a = (double*)malloc(m * n * sizeof(double))) == NULL) {
        fprintf(stderr, "error alocating memory.\n");
        fclose(m_file);
	return NULL;
    }
    if(fread(a, sizeof(double), m * n, m_file) < m * n) {
		fclose(m_file);
		return NULL;
    }
    fclose(m_file);
    return a;
}

int save_bin(char* file, double* x, int m) {
	FILE *f = fopen(file, "wb");
	if(f == NULL) return 1;
	fwrite(x, sizeof(double), m, f);
	fclose(f);
	return 0;
}

int save_result(char* filename, double* x, int m) {
	FILE* f = fopen(filename, "w");
	if(f == NULL) return 1;
	for(int i = 0; i < m; ++i) {
		fprintf(f, "%d %.16lf\n", i, x[i]);
	}
	fclose(f);
	return 0;
}

int main(int argc, char** argv) {
    
	if(argc < 10 || argc > 12) {
		fprintf(stderr, "usage: %s P n m omega N A.bin b.bin x_0.bin x_a.bin\n", argv[0]);
		return 1;
	}
	int P, n, m, N;
	if((n = atoi(argv[2])) <= 0 || (m = atoi(argv[3])) <= 0) {
		fprintf(stderr, "Dimension of the matrix must be natural numbers.\n");
		return 1;
	}
	if((P = atoi(argv[1])) <= 0) {
		fprintf(stderr, "Number of threads must be natural number.\n");
		return 1;
	}
	if((N = atoi(argv[5])) < 0) {
		fprintf(stderr, "Number of iterations can't be negative.\n");
		return 1;
	}
	double w;
	if((w = atof(argv[4])) == 0) {
		fprintf(stderr, "Omega can't be zero.\n");
		return 1;
	}
	double* a = read_matrix(argv[6], n, m);
	if(a == NULL) {
		fprintf(stderr, "Error reading from file.\n");
		return 1;
	}
	double* b = read_matrix(argv[7], n, 1);
	if(b == NULL) {
		free(a);
		fprintf(stderr, "Error reading from file.\n");
		return 1;
	}
	double* x0 = read_matrix(argv[8], m, 1);

	if(x0 == NULL) {
		free(a); free(b);
		fprintf(stderr, "Error reading from file.\n");
		return 1;
	}
	double* norms = (double*)calloc(m, sizeof(double));
	if(norms == NULL) {
		free(a); free(b); free(x0);
		fprintf(stderr, "Error allocating memory.\n");
		return 1;
	}
	double* d = (double*)malloc(m * sizeof(double));
	if(d == NULL) {
		free(a); free(b); free(x0); free(norms);
		fprintf(stderr, "Error allocating memory.\n");
		return 1;
	}
	memcpy(d, x0, m * sizeof(double));
	pthread_t* thr_idx;
    thr_struct* thr_str;
    pthread_barrier_t bar;
    if((thr_idx = (pthread_t*)malloc(P * sizeof(pthread_t))) == NULL) {
        fprintf(stderr, "Error alocating memory.\n");
		free(a); free(b); free(x0); free(norms); free(d);
        return 1;
    }
    if((thr_str = (thr_struct*)malloc(P * sizeof(thr_struct))) == NULL) {
		fprintf(stderr, "Error alocating memory.\n");
		free(a); free(b); free(x0); free(norms); free(d); free(thr_idx);
        return 1;
	}
    if(pthread_barrier_init(&bar, NULL, P)) {
		free(a); free(b); free(x0); free(norms); free(d); free(thr_idx); free(thr_str);
		fprintf(stderr, "Error creating barrier.\n");
		return 1;
	}
	struct timeval start, end;

  	gettimeofday(&start, NULL);
	for(int i = 0; i < P; ++i) {
		thr_str[i].start_row = block_first(i, P, n);
		thr_str[i].end_row = block_last(i, P, n);
		thr_str[i].start_col = block_first(i, P, m);
		thr_str[i].end_col = block_last(i, P, m);
		thr_str[i].n = n;
		thr_str[i].m = m;
		thr_str[i].N = N;
		thr_str[i].w = w;
		thr_str[i].a = a;
		thr_str[i].x = x0;
		thr_str[i].r = b;
		thr_str[i].d = d;
		thr_str[i].norms = norms;
		thr_str[i].bar_ptr = &bar;
		if(pthread_create(&thr_idx[i], NULL, thr_func, (void*) &thr_str[i])) {
            fprintf(stderr, "Error creating thread.\n");
            free(a); free(b); free(x0); free(norms); free(d);
            free(thr_str); free(thr_idx);
            return 1;
        }
	}
	for(int i = 0; i < P; ++i) {
        if(pthread_join(thr_idx[i], NULL)) {
            fprintf(stderr, "Error joining thread number %d.\n", i);
            free(a); free(b); free(x0); free(norms); free(d);
            free(thr_idx); free(thr_str);
            return 1;
        }
    }
    if(pthread_barrier_destroy(&bar)) {
	 	free(a); free(b); free(x0); free(norms); free(d); free(thr_idx); free(thr_str);
		fprintf(stderr, "Error destroying barrier.\n");
		return 1;
	}
	gettimeofday(&end, NULL);

  	printf("exec time: %lf sec\n", (double)((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)) / 1000000);

    if(save_result(argv[9], x0, m)) {
		free(a); free(b); free(x0); free(norms); free(d); free(thr_idx); free(thr_str);
		fprintf(stderr, "Error saving to file.\n");
		return 1;
	}
	if(argc == 11 && strcmp(argv[10], "bin") == 0) {
		if(save_bin("xn.bin", x0, m)) {
			free(a); free(b); free(x0); free(norms); free(d); free(thr_idx); free(thr_str);
			fprintf(stderr, "Error saving to binary file.\n");
			return 1;
		}
	}
	free(a); free(b); free(x0); free(norms); free(d); free(thr_idx); free(thr_str);
	return 0;
}
