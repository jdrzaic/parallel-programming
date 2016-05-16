#include	<stdio.h>
#include	<float.h>
#include	"cuda_auxiliary.h"

template <unsigned int block_size>
__global__ void reduction(double *in_data, double *out_data, int N, int lda, int ldb) {
	double	*data = in_data + 8 * blockIdx.x * blockDim.x + blockIdx.y * lda;

	__shared__ double smem[block_size];


	unsigned int col_ind = threadIdx.x +  8 * blockIdx.x * blockDim.x;
	if(col_ind == blockIdx.y * lda && col_ind >= N) {
		smem[threadIdx.x] = 0.0;
	} else {
		smem[threadIdx.x] = fabs(data[threadIdx.x]);
	}

	for(int i = 1; i < 8; ++i) {
		unsigned int col_ind_i = threadIdx.x +  (i + 8 * blockIdx.x) * blockDim.x;
		if(col_ind_i < N && col_ind_i != blockIdx.y * lda) {
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

		smem[threadIdx.x] = fmax(smem[threadIdx.x + 32], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 16], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 8], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 4], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 2], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 1], smem[threadIdx.x]);
	}
	
	if (threadIdx.x == 0) {
		out_data[blockIdx.x + blockIdx.y * ldb] = smem[0];
	}
}

template <unsigned int block_size>
__global__ void reduction_final(double *data, int N, int lda) {
	double *bla = data + blockDim.y * lda;
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

		smem[threadIdx.x] = fmax(smem[threadIdx.x + 32], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 16], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 8], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 4], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 2], smem[threadIdx.x]);
		smem[threadIdx.x] = fmax(smem[threadIdx.x + 1], smem[threadIdx.x]);
	}
	
	if (threadIdx.x == 0) {
		out_data[blockIdx.y * ldb] = smem[0];
	}
}

int	main(int argc, char **argv) {
	double		*dev_in_array = NULL;
	double		*dev_out_array = NULL;
	double		*hst_array = NULL;

	int			N = (1 << 20); 

	double		cpu_sum;
	double		gpu_sum;

	dim3		grid;
	dim3		block;

	double		cpu_time = 0.0;
	double		gpu_time = 0.0;


	if (argc != 2) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		goto die;
	}

	host_alloc(hst_array, double, N * sizeof(double));
	init_matrix(hst_array, N, 1, 0);

	cuda_exec(cudaMalloc(&dev_in_array, N * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_out_array, N * sizeof(double)));
	cuda_exec(cudaMemcpy(dev_in_array, hst_array, N * sizeof(double), cudaMemcpyHostToDevice));

	block.x = atoi(argv[1]);
	grid.x = ((N + block.x - 1) / block.x) / 8;

	cpu_time -= timer();
	cpu_sum = sum_array(hst_array, N);
	cpu_time += timer();

	gpu_time -= timer();
	switch (block.x) {
		case 1024:
			reduction<1024><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final<1024><<<1, block>>>(dev_out_array, grid.x);
			break;
		case  512:
			reduction< 512><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final< 512><<<1, block>>>(dev_out_array, grid.x);
			break;
		case  256: 
			reduction< 256><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final< 256><<<1, block>>>(dev_out_array, grid.x);
			break;
		case  128: 
			reduction< 128><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final< 128><<<1, block>>>(dev_out_array, grid.x);
			break;
		case   64: 
			reduction<  64><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final<  64><<<1, block>>>(dev_out_array, grid.x);
			break;
		case   32: 
			reduction<  32><<<grid.x, block>>>(dev_in_array, dev_out_array, N);
			reduction_final<  32><<<1, block>>>(dev_out_array, grid.x);
			break;
	}
		
	cudaDeviceSynchronize();
	gpu_time += timer();

	cuda_exec(cudaMemcpy(&gpu_sum, dev_out_array, sizeof(double), cudaMemcpyDeviceToHost));

	printf("CPU sum: %#.16lg\n", cpu_sum);
	printf("GPU sum: %#.16lg\n", gpu_sum);
	printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);
	printf("CPU execution time: %#.3lgs\n", cpu_time);
	printf("GPU execution time: %#.3lgs\n", gpu_time);

die:
	cuda_exec(cudaFree(dev_in_array));
	cuda_exec(cudaFree(dev_out_array));

	free(hst_array);
	
	return 0;
}
	
	
