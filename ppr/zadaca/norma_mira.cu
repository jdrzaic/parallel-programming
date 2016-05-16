#include <stdio.h>
#include <float.h>
#include "cuda_wrapper.h"


#define		BLOCK_SIZE	32

__global__ void gpu_transpose(double *A, double *B, int N, int M)
{
	__shared__ double smem[BLOCK_SIZE][BLOCK_SIZE+1];

	int ix = blockIdx.x * blockDim.x + threadIdx.x; //u stupcu
	int iy = blockIdx.y * blockDim.y + threadIdx.y; //u retku

	if(ix*M+iy < M*N){
		smem[threadIdx.x][threadIdx.y] = A[iy * N + ix];
		B[ix * M + iy] = smem[threadIdx.x][threadIdx.y];
	}

}



template <unsigned int block_size>
__global__ void reduction(double *in_data, double *out_data, int M, int N)
{
	int index_bloka = blockIdx.x + blockIdx.y * gridDim.x;
	int index_dretve_u_retku = blockIdx.x * blockDim.x + threadIdx.x;
	

	double	*data = in_data + index_bloka * blockDim.x;
	double *data_za_dijagonalu = in_data + M * blockIdx.y;

	__shared__ double smem[block_size];

	smem[threadIdx.x] = fabs(data[threadIdx.x]);

	if( index_dretve_u_retku != blockIdx.y )
		smem[threadIdx.x] +=  fabs(data_za_dijagonalu[blockIdx.y]);
	
	/*smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 1 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 2 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 3 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 4 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 5 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 6 * blockDim.x]);
	smem[threadIdx.x] = fmax(smem[threadIdx.x], data[threadIdx.x + 7 * blockDim.x]);

	__syncthreads();*/


	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 512] );
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 256] );
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 128] );
	__syncthreads();

	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x +  64] );
	__syncthreads();


	if (threadIdx.x < 32) {
		volatile double *tmp = smem;

		tmp[threadIdx.x] = fmax( tmp[threadIdx.x], tmp[threadIdx.x + 32] );
		tmp[threadIdx.x] = fmax( tmp[threadIdx.x],tmp[threadIdx.x + 16] );
		tmp[threadIdx.x] = fmax( tmp[threadIdx.x],tmp[threadIdx.x + 8] );
		tmp[threadIdx.x] = fmax( tmp[threadIdx.x],tmp[threadIdx.x + 4] );
		tmp[threadIdx.x] = fmax( tmp[threadIdx.x],tmp[threadIdx.x + 2] );
		tmp[threadIdx.x] = fmax( tmp[threadIdx.x],tmp[threadIdx.x + 1] );
	}

	if (threadIdx.x == 0){
		out_data[index_bloka] = smem[0];
		//if(blockIdx.y == 199)printf("u %d je index dretve, %d je index bloka ------ %lg\n", threadIdx.x, index_bloka, smem[0] );
	}
}		



template <unsigned int block_size>
__global__ void reduction_final(double *data, int N)
{
	//printf("tu sam1\n");
	__shared__ double smem[block_size];

	smem[threadIdx.x] = 0.0;

	for (int i = threadIdx.x; i < N; i += blockDim.x)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], data[i] );

	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 512] );
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 256] );
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x + 128] );
	__syncthreads();

	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] = fmax( smem[threadIdx.x], smem[threadIdx.x +  64] );
	__syncthreads();
	
	//printf("tu sam\n");

	if (threadIdx.x < 32) {
		//printf("tu sam\n");
		volatile double *tmp = smem;

		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 32] );
		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 16] );
		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 8] );
		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 4] );
		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 2] );
		tmp[threadIdx.x] = fmax( smem[threadIdx.x], tmp[threadIdx.x + 1] );
	}

	if (threadIdx.x == 0){
		data[0] = smem[0];
		//printf("konacno rjesenje je %lg\n", smem[0]);
	}
}

void cpu_transpose(double *hst_A, double *hst_B, int N, int M)
{
	/*for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j)
			hst_B[j * N + i] = hst_A[j];

		hst_A += M;
	}*/

	for(int i = 0; i < N; ++i){
		for(int j = 0; j < M; ++j){
			hst_B[i*M+j] = hst_A[j*N+i];
		}
	}
}

void check_result(double *cpu_result, double *gpu_result, int N, int M)
{
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j){
			if (abs(cpu_result[i * M + j] - gpu_result[i * M + j] )> 0.001 ) {
				printf("GPU and CPU results differ at position %d %d\n", i, j);
				return ;
			}
		}
	}

	printf("GPU transpose is correct\n");
}

double max_array(double *data, int N, int M)
{
	double	maxi=0, sum;

	for(int i=0; i < N*M; ++i){
		if(i%M>=N)
			continue;
		if(i/M == i%M)
			continue;
		sum = abs(data[i])+abs(data[(i/M)*M + i/M]);
		if(sum > maxi){
			maxi = sum;
		}
	}
	return maxi;
}

void ispis( double *polje, int N ){

	for(int j = 0; j < N; ++j ){
		for( int i = 0; i < N; ++i )
			printf("%lg ", polje[N*i +j] );
		printf("\n");
	}		
}


void nadopuni_nulama(double *stari,double *novi, int N, int M){

	for(int i = 0; i < M; ++i)
		for(int j = 0; j < M; ++j)
			if(j<N && i < N)
				novi[j*M+i] = stari[j*N+i];
			else
				novi[j*M+i] = 0;
	
}

int	main(int argc, char **argv)
{

	double		*dev_in_array = NULL;
	double		*dev_out_array = NULL;
	double		*hst_array = NULL;
	double 		*hst_help = NULL;
	double 		*hst_print = NULL;


	int			N, M;

	dim3		grid;
	dim3		block;

	double		cpu_time = 0.0;
	double		gpu_time = 0.0;

	double      maxi_gpu;
	double      maxi_cpu;

	size_t koliko;

	if (argc != 3 ) {
		fprintf(stderr, "usage: %s dimx\n", argv[0]);
		goto die;
	}
	
	N = atoi( argv[1] );

   	///alociramo memoriju za device
	host_alloc(hst_array, double, N * N * sizeof(double));

	///ucitamo matricu
	FILE *f;
	f = fopen(argv[2], "r");
   	if( f == NULL ){
        	fprintf(stderr, "Greska pri otvaranju datoteke\n");
       		 exit( EXIT_FAILURE );
   	}
    fread( hst_array, sizeof(double), N * N, f );

    //ideja: tu cemo dio memorije popuniti nulama da se kod transponiranja nasa matrica u retcima nadopuni nulama do visekratnika 128
    M = 128 - ( N % 128 );
    if(M == 128)
    	M = 0;
   	koliko = sizeof(double) * M;
    M = N + M;
  
    cudaMemset(&hst_array[N*N], 0, koliko);

	host_alloc(hst_help, double, N * M * sizeof(double));
	host_alloc(hst_print, double, N * M * sizeof(double));


	cuda_exec(cudaMalloc(&dev_in_array, N * M * sizeof(double)));
	cuda_exec(cudaMalloc(&dev_out_array, N * M * sizeof(double)));
	cuda_exec(cudaMemcpy(dev_in_array, hst_array, N * M * sizeof(double), cudaMemcpyHostToDevice));	


	cpu_transpose(hst_array, hst_help, N, M);

	block.x = BLOCK_SIZE;
	block.y = BLOCK_SIZE;

	grid.x = (N + block.x - 1) / block.x;
	grid.y = (M + block.x -1) / block.x;

	//printf("%d je grid.x, %d je grid.y\n", grid.x, grid.y);
	gpu_time -= timer();
	gpu_transpose<<<grid, block>>>(dev_in_array, dev_out_array, N, M);	
	cuda_exec(cudaDeviceSynchronize());
	gpu_time += timer();
	cuda_exec(cudaMemcpy(hst_array, dev_out_array, N * M * sizeof(double), cudaMemcpyDeviceToHost));

	check_result(hst_help, hst_array, N, M);

	cuda_exec(cudaMemcpy(dev_in_array, hst_array, N * M * sizeof(double), cudaMemcpyHostToDevice));

	 // for(int i = 128*31; i< 128*32	; ++i)
	 // 	printf("%lg cpu <----> %lg gpu ..... %d\n", hst_help[i], hst_array[i], i);


	cpu_time -= timer();
	maxi_cpu = max_array(hst_array, N, M);
	cpu_time += timer();

	block.x = 128;
	block.y = 1;
	grid.x = ((M + block.x - 1) / block.x);
	grid.y = N;

	printf("Execution configuration: %d blocks, %d threads\n", grid.x*grid.y, block.x);

	gpu_time -= timer();

	reduction< 128><<<grid, block>>>(dev_in_array, dev_out_array, M, N);
	reduction_final< 128><<<1, block>>>(dev_out_array, grid.x*grid.y);

	cudaDeviceSynchronize();
	gpu_time += timer();

	cuda_exec(cudaMemcpy(&maxi_gpu, dev_out_array, sizeof(double), cudaMemcpyDeviceToHost));

	printf("CPU max: %#.16lg\n", maxi_cpu);
	printf("GPU max: %#.16lg\n", maxi_gpu);
	//printf("Execution configuration: %d blocks, %d threads\n", grid.x, block.x);
	printf("CPU execution time: %#.3lgs\n", cpu_time);
	printf("GPU execution time: %#.3lgs\n", gpu_time);

die:
	cuda_exec(cudaFree(dev_in_array));
	cuda_exec(cudaFree(dev_out_array));


	free(hst_help);
	free(hst_array);
	free(hst_print);

	return 0;
}






