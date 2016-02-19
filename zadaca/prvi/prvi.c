#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#define block_first(i,T,N)		(((i) * (N)) / (T))
#define block_last(i,T,N)		(block_first(i+1,T,N) - 1)
#define block_size(i,T,N)		(block_first(i+1,T,N) - block_first(i,T,N))

typedef struct {
    int start;
    int len;
    char* a;
    int n;
    pthread_mutex_t* mutex;
} thr_struct;

void* thr_func(void* arg) {
    int start = ((thr_struct*)arg)->start;
    char* a = ((thr_struct*)arg)->a;
    int n = ((thr_struct*)arg)->n;
    pthread_mutex_t* mutex = ((thr_struct*)arg)->mutex;
    int len = ((thr_struct*)arg)->len;
    for(int k = 0; k < len; ++k) {
        if(a[k] == 'o') {
            int i = (k + start) / n;
            int j = (k + start) % n;
            pthread_mutex_lock(mutex);
            printf("(%d, %d)\n", i, j);
       	   pthread_mutex_unlock(mutex);
        }
    }
}

int main(int argc, char **argv) {
    int P, M, N;
    pthread_t* thr_idx;
    thr_struct* thr_str;
    pthread_mutex_t mutex;
    
    if(argc != 5) {
        fprintf(stderr, "usage: %s P M N a.txt\n", argv[0]);
	return 1;
    }
    if((P = atoi(argv[1])) <= 0 || (M = atoi(argv[2])) <= 0 || (N = atoi(argv[3])) <= 0) {
        fprintf(stderr, "Number of threads and both dimensions of the board must be positive integers.\n");
        return 1;
    }
    char* a;
    if((a = (char*)malloc((M * N + 1) * sizeof(char))) == NULL) {
        fprintf(stderr, "Error alocating memory.\n");
        return 1;
    }
    a[0] = '\0';
    FILE* file =fopen(argv[4], "r");
    if(file) {
	char *tmp = a;
        while(1) {
            if(fscanf(file, "%s", tmp) <= 0) {
                break;
            }
	    tmp += strlen(tmp);
        }
    } else {
        fprintf(stderr, "error opening file.\n");
        free(a);
        return 1;
    }
    if((thr_idx = (pthread_t*)malloc(P * sizeof(pthread_t))) == NULL) {
        fprintf(stderr, "Error alocating memory.\n");
        free(a);
        return 1;
    }
    if ((thr_str = (thr_struct*)malloc(P * sizeof(thr_struct))) == NULL) {
        fprintf(stderr, "Error alocating memory.\n");
	free(thr_idx);
        free(a);
        return 1;
    }

    pthread_mutex_init(&mutex, NULL);
    for(int i = 0; i < P; ++i) {
        thr_str[i].len = block_size(i, P, M * N);
        thr_str[i].n = N;
        int start = block_first(i, P, M * N);
        thr_str[i].start = start;
        thr_str[i].a = a + start;
        thr_str[i].mutex = &mutex;
        if(pthread_create(&thr_idx[i], NULL, thr_func, (void*) &thr_str[i])) {
            fprintf(stderr, "Error creating thread.\n");
            free(a);
            free(thr_str);
            free(thr_idx);
            return 1;
        }
    }

    for(int i = 0; i < P; ++i) {
        if(pthread_join(thr_idx[i], NULL)) {
            fprintf(stderr, "Error joining thread number %d.\n", i);
            free(a);
            free(thr_idx);
            free(thr_str);
            return 1;
        }
    }
    pthread_mutex_destroy(&mutex);
    free(a);
    free(thr_idx);
    free(thr_str);
    return 0;
}
