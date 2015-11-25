#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>

#define block_first(i,T,N)		(((i) * (N)) / (T))
#define block_last(i,T,N)		(block_first(i+1,T,N) - 1)
#define block_size(i,T,N)		(block_first(i+1,T,N) - block_first(i,T,N))

extern void dger_(int* M, int* N, double* ALPHA, double* X, int* INCX, double* Y, int* INCY,
    double* A, int* LDA);

typedef struct {
    int idx;
    int ntr;
    int m;
    int n;
    int p;

    double* a;
    double* b;
    double* c;
} thr_struct;

void* thr_func(void* arg) {
    int idx = ((thr_struct*)arg)->idx;
    int ntr = ((thr_struct*)arg)->ntr;
    int m = ((thr_struct*)arg)->m;
    int n = ((thr_struct*)arg)->n;
    int p = ((thr_struct*)arg)->p;

    double* a = ((thr_struct*)arg)->a;
    double* b = ((thr_struct*)arg)->b;
    double* c = ((thr_struct*)arg)->c;

    int inc = 1;
    double alpha = 1.0;
    for(int i = 0; i < n; ++i) {
        dger_(&m, &p, &alpha, a + i * m, &inc, b + i, &n, c, &m);
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

int save_result(char* file, double* a, int m, int n) {
    FILE* wfile=fopen(file,"wb");
    if(!wfile){
        return 0;
    }
    
    fwrite(a, sizeof(double), m * n, wfile);
    
    fclose(wfile);
    return 1;
}

int main(int argc, char** argv) {
    pthread_t* thr_idx;
    thr_struct* thr_str;
    int thr_num;

    if(argc != 8) {
        fprintf(stderr, "usage: %s t_num m n p A.dat B.dat C.dat\n", argv[0]);
        return 1;
    }

    if((thr_num = atoi(argv[1])) <= 0) {
        fprintf(stderr, "%s: number of threads must be positive.\n", argv[0]);
	return 1;
    }
    //dimensions
    int m, n, p;
    if((m = atoi(argv[2])) <= 0 || (n = atoi(argv[3])) <= 0 || (p = atoi(argv[4])) <= 0) {
        fprintf(stderr, "%s: dimensions must be positive.\n", argv[0]);
	return 1;
    }

    //load matrices
    double* a = NULL;
    double* b = NULL;
    double* c = (double*)calloc(m * p, sizeof(double));
    if((a = read_matrix(argv[5], m, n)) == NULL ||
        (b = read_matrix(argv[6], n, p)) == NULL) {
        printf("unable to read from input files.\n");
        return 1;
    }

    if((thr_idx = (pthread_t*)malloc(thr_num * sizeof(pthread_t))) == NULL ||
        (thr_str = (thr_struct*)malloc(thr_num * sizeof(thr_struct))) == NULL) {
        fprintf(stderr, "%s: error alocating memory.\n", argv[0]);
	free(a);
	free(b);
	free(c);
        return 1;
    }

    for(int i = 0; i < thr_num; ++i) {
        thr_str[i].idx = i;
        thr_str[i].ntr = thr_num;
        thr_str[i].m = m;
        thr_str[i].n = n;
        thr_str[i].p = block_size(i, thr_num, p);
        thr_str[i].a = a;
        thr_str[i].b = b + block_first(i, thr_num, p) * n;
        thr_str[i].c = c + block_first(i, thr_num, p) * m;

        if(pthread_create(&thr_idx[i], NULL, thr_func, (void*) &thr_str[i])) {
            fprintf(stderr, "error creating thread.\n");
	    free(a);
	    free(b);
	    free(c);
	    free(thr_idx);
	    free(thr_str);
            return 1;
        }
    }

    for(int i = 0; i < thr_num; ++i) {
        if(pthread_join(thr_idx[i], NULL)) {
            fprintf(stderr, "error joining thread number %d.\n", i);
	    free(a);
	    free(b);
	    free(c);
	    free(thr_idx);
	    free(thr_str);
            return 1;
        }
    }

    if(!save_result(argv[7], c, m, p)) {
        fprintf(stderr, "Error saving result to file %s.\n", argv[7]);
    }
    free(a);
    free(b);
    free(c);
    free(thr_idx);
    free(thr_str);
    return 0;
}
