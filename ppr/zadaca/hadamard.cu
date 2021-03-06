#include "cuda_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

unsigned int BLOCK_SIZE_X = 512;
unsigned int BLOCK_SIZE_Y = 1;

__global__ void hadamard_mul(double* A, double* B, double* C, const int lda, const int nx, const int ny) {
	unsigned int x_ind = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_ind = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int ind = y_ind * lda + x_ind;

	if(x_ind < nx && y_ind < ny) {
		C[ind] = A[ind] * B[ind];
	}
}

double* read_matrix(char* file, int m, int n) {
    FILE* m_file = fopen(file, "rb");
    if(!m_file) {
        exit(1);
    }
    double *a;
    if((a = (double*)malloc(m * n * sizeof(double))) == NULL) {
        fprintf(stderr, "error alocating memory.\n");
        fclose(m_file);
		exit(1);
    }
    if(fread(a, sizeof(double), m * n, m_file) < m * n) {
		fprintf(stderr, "error reading file %s.\n", file);
		fclose(m_file);
		exit(1);
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

void hadamard_mul_cpu(int m, int n, double* A, double* B) {
	for(int i = 0; i < m * n; ++i) {
		B[i] = A[i] * B[i];
	}
}

int check_result(int m, int n, double* cpu_res, double* gpu_res) {
	for(int i = 0; i < m * n; ++i) {
		if(cpu_res[i] != gpu_res[i]) {
			printf("%d %lf %lf\n", i , cpu_res[i], gpu_res[i]);
			return 1;
		}
	}
	printf("ok\n");
	return 0;
}

int main(int argc, char** argv) {

	if(argc != 6) {
		printf("usage: %s M N A.dat B.dat C.dat\n", argv[0]);
		return 1;
	} 

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);

	double* hst_A = NULL;
	double* hst_B = NULL;
	double* hst_C = NULL;

	double* dev_A = NULL;
	double* dev_B = NULL;
	double* dev_C = NULL;

	double cpu_time = 0;
	double gpu_time = 0;
	
	hst_A = read_matrix(argv[3], m, n);
	hst_B = read_matrix(argv[4], m, n);
	hst_C = (double*)malloc(n * m * sizeof(double));

	size_t pitch;
	int lda; 

	dim3 block_size;
	dim3 grid_size;

	cuda_exec(cudaMallocPitch(&dev_A, &pitch, m * sizeof(double), n));
	cuda_exec(cudaMallocPitch(&dev_B, &pitch, m * sizeof(double), n));
	cuda_exec(cudaMallocPitch(&dev_C, &pitch, m * sizeof(double), n));

	lda = pitch / sizeof(double);

	cuda_exec(cudaMemcpy2D(dev_A, pitch, hst_A, m * sizeof(double), m * sizeof(double), n,
		cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy2D(dev_B, pitch, hst_B, m * sizeof(double), m * sizeof(double), n,
		cudaMemcpyHostToDevice));

	block_size.x = BLOCK_SIZE_X;
	block_size.y = BLOCK_SIZE_Y;

	grid_size.x = min((m + block_size.x - 1) / block_size.x, 65535);
	grid_size.y = min((n + block_size.y - 1) / block_size.y, 65535);

	gpu_time -= timer();
	hadamard_mul<<<grid_size, block_size>>>(dev_A, dev_B, dev_C, lda, m, n);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy2D(hst_C, m * sizeof(double), dev_C, pitch, m * sizeof(double), n, 
		cudaMemcpyDeviceToHost));

	cpu_time -= timer();
	hadamard_mul_cpu(m, n, hst_A, hst_B);
	cpu_time += timer();

	check_result(m, n, hst_B, hst_C);
	
	save_bin(argv[5], hst_C, m * n); 

	printf("gpu time: %lf, cpu time: %lf", gpu_time, cpu_time);
	
	free(hst_A);
	free(hst_B);
	free(hst_C);
	
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	
	return 0;
}
