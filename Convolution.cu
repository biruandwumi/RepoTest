#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

using namespace std;
/**
 * CUDA Kernel Device code
 *
 * Computes the max_pooling of A into C. 
 */

__global__ void Convolution(const float *A, const float *K, float *C, int in_rows, int in_cols, int out_rows, int out_cols, int stride, int kernel_size, int padding)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	// policy = 0: same
	// policy = 1: valid
	int valid_offset = (padding == 1) ? kernel_size / 2 : 0;
	
	if (i < out_rows && j < out_cols){
        int offset_i = i * stride;
		int offset_j = j * stride;
		float sum = 0;

		for(int k_row_index = -kernel_size/2; k_row_index <= kernel_size/2; k_row_index++){
			for(int k_col_index = -kernel_size/2; k_col_index <= kernel_size/2; k_col_index++){
				int in_row_index =  offset_i + k_row_index + valid_offset;
				int in_col_index =  offset_j + k_col_index + valid_offset;
				if(in_row_index < 0 || in_row_index >= in_rows || in_col_index < 0 || in_col_index >= in_cols)
					continue; // only happens for same mode, skipping is equivalent to summing with padded zeros
				sum += A[in_row_index * in_cols + in_col_index] * K[(k_row_index + kernel_size/2)*kernel_size + (k_col_index + kernel_size/2)];
			}
		}
		C[i * out_cols + j] = sum;
		
	}

}

void HelperOutputDim(int in_row, int in_col, int stride, int kernel_size, int &out_row, int &out_col, int padding_policy){
	
	switch(padding_policy){
		case 0: // any / same
			out_row = (int)ceil((float)in_row / (float)stride);
			out_col = (int)ceil((float)in_col / (float)stride);
			break;
		case 1: // valid
			out_row = (int)ceil((float)(in_row - kernel_size + 1) / (float)stride);
			out_col = (int)ceil((float)(in_col - kernel_size + 1) / (float)stride);
			break;
		default:
			break;
	}
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	int in_rows = 5;
	int in_cols = 5;
	int kernel_size = 3;
	int stride = 1;
	int padding_policy = 1;
	size_t size_K = kernel_size * kernel_size* sizeof(float);
    size_t size_A = in_rows * in_cols* sizeof(float);
    printf("[Matrix of %d elements]\n", in_rows * in_cols);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size_A);

	// Allocate the host input vector K
    float *h_K = (float *)malloc(size_K);

    // Allocate the host output vector C
	int out_rows = 0;
	int out_cols = 0;
	HelperOutputDim(in_rows, in_cols, stride, kernel_size, out_rows, out_cols, padding_policy);
	size_t size_C = out_rows * out_cols* sizeof(float);
    float *h_C = (float *)malloc(size_C);

    // Verify that allocations succeeded
    if (h_A == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
	for(int i = 0; i < in_rows; i++){
		for(int j = 0; j < in_cols; j++){
			h_A[i * in_cols + j] = 1; //i * in_cols + j;
		}
	}
	 // Initialize the host input vectors
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			h_K[i * kernel_size + j] = 2;
		}
	}

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Allocate the device input vector K
    float *d_K = NULL;
    err = cudaMalloc((void **)&d_K, size_K);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector K (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  
    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and K in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	 // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector K from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Launch the MaxPooling CUDA Kernel
	dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(in_rows / threadsPerBlock.x + 1, in_cols / threadsPerBlock.y + 1);
    Convolution<<<numBlocks, threadsPerBlock>>>(d_A, d_K, d_C, in_rows, in_cols, out_rows, out_cols, stride, kernel_size, padding_policy);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    
	for(int i = 0; i < out_rows; i++){
		for(int j = 0; j < out_cols; j++){
			printf("%f ",h_C[i*out_cols+j]);
		}
		printf("\n");
	}

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaFree(d_K);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector K (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
	free(h_K);
    free(h_C);

    printf("Done\n");
    return 0;
}
