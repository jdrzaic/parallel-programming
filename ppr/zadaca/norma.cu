#include "cuda_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_X = 512;
const int BLOCK_SIZE_Y = 1;

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

__global__ void gpu_transpose(double* A, double* B, int N, int lda) {
	__shared__ double smem[BLOCK_SIZE][BLOCK_SIZE+1];

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(ix < N && iy < N) {
		smem[threadIdx.x][threadIdx.y] = A[iy * lda + ix];
		B[ix * lda + iy] = smem[threadIdx.x][threadIdx.y];
	}
}

__device__ void print(int block_size, double *data) {
    return;
	if (threadIdx.x == 0 && blockIdx.y == 0) {
		for (int i = 0; i < 10; ++i) {
			printf("%lf ", data[i]);
		}
		printf("\n");	
	}
}

template <unsigned int block_size>
__global__ void reduction(double *in_data, double *out_data, int N, int lda) {
	double	*data = in_data + 8 * blockIdx.x * blockDim.x + blockIdx.y * lda;

	__shared__ double smem[block_size];


	unsigned int col_ind = threadIdx.x +  8 * blockIdx.x * blockDim.x;
	if(col_ind == blockIdx.y || col_ind >= N) {
		smem[threadIdx.x] = 0.0;
	} else {
		smem[threadIdx.x] = fabs(data[threadIdx.x]);
	}

	for(int i = 1; i < 8; ++i) {
		unsigned int col_ind_i = threadIdx.x +  (i + 8 * blockIdx.x) * blockDim.x;
		if(col_ind_i < N && col_ind_i != blockIdx.y) {
			smem[threadIdx.x] = fmax(fabs(data[threadIdx.x + i * blockDim.x]), smem[threadIdx.x]);
		}
	}	
	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 512], smem[threadIdx.x]);
	__syncthreads();

	
	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 256], smem[threadIdx.x]);
	__syncthreads();
	
	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 128], smem[threadIdx.x]);
	__syncthreads();
	
	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 64], smem[threadIdx.x]);
	__syncthreads();
	
		
	if (threadIdx.x < 32) {
		volatile double *tmp = smem;
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 32], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 16], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 8], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 4], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 2], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 1], tmp[threadIdx.x]);
	}
	
	if (threadIdx.x == 0) {
		//__syncthreads();
		//printf("%d %d %lf\n", blockIdx.x, blockIdx.y, smem[0]);
		out_data[blockIdx.x + blockIdx.y * lda] = smem[0];
	}
}

template <unsigned int block_size>
__global__ void reduction2(double *max_rows, double *in_data, int N, int lda) {
	double *data = in_data + 8 * blockIdx.x * blockDim.x + 8 * blockIdx.x * blockDim.x * lda;
    double *mr = max_rows + 8 * blockIdx.x * blockDim.x * lda;

	__shared__ double smem[block_size];

	unsigned int col_ind = threadIdx.x +  8 * blockIdx.x * blockDim.x;
	
	if(col_ind >= N) {
		smem[threadIdx.x] = 0.0;
	} else {
		smem[threadIdx.x] = fabs(data[threadIdx.x + lda*threadIdx.x]) + mr[threadIdx.x * lda];
	}

	for(int i = 1; i < 8; ++i) {
		unsigned int col_ind_i = threadIdx.x +  (i + 8 * blockIdx.x) * blockDim.x;
		if(col_ind_i < N) {
			smem[threadIdx.x] = fmax(fabs(data[(threadIdx.x + i * blockDim.x) * (lda + 1)]) +
			mr[(threadIdx.x + blockDim.x * i) * lda], smem[threadIdx.x]);
		}
	}	
	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 512], smem[threadIdx.x]);
	__syncthreads();

	
	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 256], smem[threadIdx.x]);
	__syncthreads();
	
	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 128], smem[threadIdx.x]);
	__syncthreads();
	
	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 64], smem[threadIdx.x]);
	__syncthreads();
	
		
	if (threadIdx.x < 32) {
		volatile double *tmp = smem;
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 32], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 16], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 8], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 4], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 2], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 1], tmp[threadIdx.x]);
	}
	
	if (threadIdx.x == 0) {
		//__syncthreads();
		//printf("%d %d %lf\n", blockIdx.x, blockIdx.y, smem[0]);
		in_data[blockIdx.x] = smem[0];
	}
}

template <unsigned int block_size>
__global__ void reduction_final(double *data, int N, int lda) {
	double *bla = data + blockIdx.y * lda;
	__shared__ double smem[block_size];

	smem[threadIdx.x] = 0.0;

	for (int i = threadIdx.x; i < N; i += blockDim.x) {
		smem[threadIdx.x] = fmax(smem[threadIdx.x], bla[i]);
	}

	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 512], smem[threadIdx.x]);
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 256], smem[threadIdx.x]);
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 128], smem[threadIdx.x]);
	__syncthreads();

	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 64], smem[threadIdx.x]);
	__syncthreads();

		
	if (threadIdx.x < 32) {
		volatile double *tmp = smem;

		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 32], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 16], tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 8],  tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 4],  tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 2],  tmp[threadIdx.x]);
		tmp[threadIdx.x] = fmax(tmp[threadIdx.x + 1],  tmp[threadIdx.x]);
	}
	
	if (threadIdx.x == 0) {
		//printf("%d %lf \n", blockIdx.y, smem[0]);
		data[blockIdx.y * lda] = smem[0];
	}
}

int main(int argc, char** argv) {
	
	if(argc != 3) {
		printf("usage: %s N A.dat\n", argv[0]);
		return 1;
	} 

	int N = atoi(argv[1]);
	
	double* hst_A = NULL;
	double* dev_A = NULL;
	double* dev_B = NULL;

	double cpu_time = 0;
	double gpu_time = 0;
	
	hst_A = read_matrix(argv[2], N, N);

	size_t pitch;
	int lda; 

	dim3 block_size;
	dim3 grid_size;

	cuda_exec(cudaMallocPitch(&dev_A, &pitch, N * sizeof(double), N));
	cuda_exec(cudaMallocPitch(&dev_B, &pitch, N * sizeof(double), N));
	
	lda = pitch / sizeof(double);

	cuda_exec(cudaMemcpy2D(dev_A, pitch, hst_A, N * sizeof(double), N * sizeof(double), N,
		cudaMemcpyHostToDevice));

    /*
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf(" %lf", hst_A[i*N + j]);
		}
		printf("\n");
	}
    */
	
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;

	grid_size.x = ((N + block_size.x - 1) / (block_size.x));
	grid_size.y = grid_size.x;

	gpu_time -= timer();
	gpu_transpose<<<grid_size, block_size>>>(dev_A, dev_B, N,lda );	

	cudaMemcpy2D(hst_A, N*sizeof(double), dev_B, pitch, N * sizeof(double), N,
				cudaMemcpyDeviceToHost);
/*	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf(" %lf", hst_A[i*N + j]);
		}
		printf("\n");
	}
    */
	block_size.x = BLOCK_SIZE_X;
	block_size.y = BLOCK_SIZE_Y;

	grid_size.x = ((N + 8 * block_size.x - 1) / (8 * block_size.x));
	grid_size.y = N;

	reduction<BLOCK_SIZE_X><<<grid_size, block_size>>>(dev_B, dev_A, N, lda);
    int nn = grid_size.x;
	grid_size.x = 1;
	reduction_final<BLOCK_SIZE_X><<<grid_size, block_size>>>(dev_A, nn, lda);
	
	reduction2<BLOCK_SIZE_X><<<nn, block_size>>>(dev_A, dev_B, N, lda);
    reduction_final<BLOCK_SIZE_X><<<1, block_size>>>(dev_B, nn, lda);
	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();

	cuda_exec(cudaMemcpy2D(hst_A, N * sizeof(double), dev_B, pitch, N * sizeof(double), N, 
		cudaMemcpyDeviceToHost));	
	
	printf("%lf\n", hst_A[0]);
	printf("gpu time: %lf\n", gpu_time);
	
	free(hst_A);
	
	cudaFree(dev_A);
	cudaFree(dev_B);
	
	return 0;
	
}
