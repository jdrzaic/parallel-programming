#include <stdio.h>
#include <stdlib.h>
#include "../cuda_wrapper.h"

typedef int vertex;

const int BLOCK_SIZE = 256;
const int EDGES = 8;

typedef struct {
	vertex* v;
	vertex* w;
} graph;

__global__ void label_new(graph g, vertex* will_visit, vertex* visited, vertex r, int N) {
	int threads = gridDim.x * blockDim.x;
	int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = thr_idx; i < N; i += threads) {
		printf("index dretve %d\n", thr_idx);
		printf("index = %d, first = %d, second = %d, visited = %d, will_visit = %d\n", i, (g.v)[i], (g.w)[i], visited[(g.v)[i]],will_visit[(g.w)[i]] );
		if(visited[(g.v)[i]] && !visited[(g.w)[i]]) {			will_visit[(g.w)[i]] = 1;
			printf("will visit %d \n", (g.w)[i]);
		}
	}
}

template <unsigned int block_size>
__global__ void reduction(int* in_data, int* visited, int* out_data, int N)
{
	int* data = in_data + 8 * blockIdx.x * blockDim.x;
	__shared__ int smem[block_size];

	int thr_idx = 8 * blockIdx.x * blockDim.x + threadIdx.x;
	if(thr_idx < N) {
		smem[threadIdx.x] = data[threadIdx.x];
		if(data[threadIdx.x]) {
			visited[thr_idx] = 1;
		}
	} else {
		smem[threadIdx.x] = 0;
	}
	#pragma unroll
	for(int i = 1; i < 8; ++i) {
		int thr_idx_i = thr_idx + i * blockDim.x;
		if(thr_idx_i >= N) break;
		smem[threadIdx.x] += data[threadIdx.x + i * blockDim.x];
		if(data[threadIdx.x + i * blockDim.x]) {
			visited[thr_idx_i] = 1;
		}
 	}
	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] += smem[threadIdx.x + 512];
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] += smem[threadIdx.x + 256];
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] += smem[threadIdx.x + 128];
	__syncthreads();

	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] += smem[threadIdx.x +  64];
	__syncthreads();

		
	if (threadIdx.x < 32) {
		volatile int *tmp = smem;

		tmp[threadIdx.x] += tmp[threadIdx.x + 32];
		tmp[threadIdx.x] += tmp[threadIdx.x + 16];
		tmp[threadIdx.x] += tmp[threadIdx.x +  8];
		tmp[threadIdx.x] += tmp[threadIdx.x +  4];
		tmp[threadIdx.x] += tmp[threadIdx.x +  2];
		tmp[threadIdx.x] += tmp[threadIdx.x +  1];
	}
	
	if (threadIdx.x == 0) {
		out_data[blockIdx.x] = smem[0];
		printf("first reduction %d\n", smem[0]);
	}
}

template <unsigned int block_size>
__global__ void reduction_final(int *data, int N)
{
	__shared__ int smem[block_size];

	smem[threadIdx.x] = 0.0;

	for (int i = threadIdx.x; i < N; i += blockDim.x) {
		smem[threadIdx.x] += data[i];
	}

	__syncthreads();

	if (block_size >= 1024 && threadIdx.x < 512)
		smem[threadIdx.x] += smem[threadIdx.x + 512];
	__syncthreads();

	if (block_size >=  512 && threadIdx.x < 256)
		smem[threadIdx.x] += smem[threadIdx.x + 256];
	__syncthreads();

	if (block_size >=  256 && threadIdx.x < 128)
		smem[threadIdx.x] += smem[threadIdx.x + 128];
	__syncthreads();

	if (block_size >=  128 && threadIdx.x <  64)
		smem[threadIdx.x] += smem[threadIdx.x +  64];
	__syncthreads();

		
	if (threadIdx.x < 32) {
		volatile int *tmp = smem;

		tmp[threadIdx.x] += tmp[threadIdx.x + 32];
		tmp[threadIdx.x] += tmp[threadIdx.x + 16];
		tmp[threadIdx.x] += tmp[threadIdx.x +  8];
		tmp[threadIdx.x] += tmp[threadIdx.x +  4];
		tmp[threadIdx.x] += tmp[threadIdx.x +  2];
		tmp[threadIdx.x] += tmp[threadIdx.x +  1];
	}
	
	if (threadIdx.x == 0) {
		data[0] = smem[0];
		printf("number of new vertices = %d\n", data[0]);
	}
}

int main(int argc, char** argv) {

	if(argc != 3) {
		printf("Path to file missing\n");
		return 0;
	}
	graph hst_g;
	graph dev_g;

	vertex* hst_visited;
	vertex* dev_visited;
	
	vertex* hst_will_visit;
	vertex* dev_will_visit;

	vertex* hst_out;
	vertex* dev_out;

	double vertices = 0;
	double last_vertices;

	dim3 block_size;
	dim3 grid_size;

	dim3 block_size_red;
	dim3 grid_size_red;

	FILE* f;

	int V;
	int E;

	f = fopen(argv[1], "r");
	
	fscanf(f, "%d", &V);
	fscanf(f, "%d", &E);

	hst_g.v = (vertex*)malloc(2 * E * sizeof(vertex));
	hst_g.w = hst_g.v + E;
	cuda_exec(cudaMalloc(&dev_g.v, 2 * E * sizeof(vertex)));
	dev_g.w = dev_g.v + E;
	hst_visited = (vertex*)calloc(V, sizeof(vertex));
	cuda_exec(cudaMalloc(&dev_visited, V * sizeof(vertex)));
	hst_will_visit = (vertex*)calloc(V, sizeof(vertex));
	cuda_exec(cudaMalloc(&dev_will_visit, V * sizeof(vertex)));
	hst_out = (vertex*)calloc(V, sizeof(vertex));
	cuda_exec(cudaMalloc(&dev_out, V * sizeof(vertex)));
	
	//sortirani rastuci po indeksu prvog cvora
	int j;
	vertex r = -1;
	int tren = 0;
	int max = 0;
	vertex read;
	for(int i = 0; i < V; ++i) {
		while(1) {
			fscanf(f, "%d", &read);
			if(read == -1) {
				if(tren > max) {
					max = tren;
					r = i;
					tren = 0;
				}
				break;
			}
			++tren;
			(hst_g.v)[j] = i;
			(hst_g.w)[j] = read;
			printf("%d %d %d\n", j,(hst_g.v)[j], (hst_g.w)[j]);
			++j;
		}
	}
	printf("r : %d\n", r);
	hst_visited[r] = 1;
	cuda_exec(cudaMemcpy(dev_visited, hst_visited, V * sizeof(vertex), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_g.v, hst_g.v, 2 * E * sizeof(vertex), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_will_visit, hst_will_visit, V * sizeof(vertex), cudaMemcpyHostToDevice));
	cuda_exec(cudaMemcpy(dev_out, hst_out, V * sizeof(vertex), cudaMemcpyHostToDevice));
	block_size.x = BLOCK_SIZE;
	grid_size.x = ((E + EDGES * block_size.x - 1) / (EDGES * block_size.x));
	block_size_red.x = BLOCK_SIZE;
	grid_size_red.x = ((V + 8 * block_size_red.x - 1) / (8 * block_size_red.x));
	while(vertices < V / 2) {
		label_new<<<grid_size.x, block_size.x>>>(dev_g, dev_will_visit, dev_visited, r, E);
		reduction<BLOCK_SIZE><<<grid_size_red.x, block_size_red.x>>>(dev_will_visit, dev_visited, dev_out, V);
		reduction_final<BLOCK_SIZE><<<1, block_size_red.x>>>(dev_out, grid_size_red.x);
		cuda_exec(cudaMemcpy(&last_vertices, dev_out, sizeof(int), cudaMemcpyDeviceToHost));
		printf("novih host %d\n",last_vertices);
		vertices += last_vertices;
		printf("vertices %d\n", vertices);
		printf("grid size = %d", grid_size_red.x);
	}
	int remove_last = 0;
	if(vertices - V / 2 > V / 2 - vertices + last_vertices) {
		remove_last = 1;
	}
	cuda_exec(cudaMemcpy(&hst_visited, dev_visited, V * sizeof(int), cudaMemcpyDeviceToHost));
	cuda_exec(cudaMemcpy(&hst_will_visit, dev_will_visit, V * sizeof(int), cudaMemcpyDeviceToHost));
	FILE* of = fopen(argv[2], "w");
	for(int i = 0; i < V; ++i) {
		if(!hst_visited[i]) {
			continue;
		} else if(hst_will_visit[i] && !remove_last || !hst_will_visit[i]) {
			fprintf(of, "%d\n", i);
		}
	} 

	return 0;
}
