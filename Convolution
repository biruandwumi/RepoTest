#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

float* AllocateTempBuffer(int n_rows, int n_cols){
	float *matrix = (float*)malloc(n_rows * n_cols * sizeof(float*));
	// initialize to 0
	for(int i = 0; i < n_rows; i++){
		for(int j = 0; j < n_cols; j++){
			matrix[i * n_cols + j] = 0;
		}
	}
	return matrix;
}

void DeAllocateTempBuffer(float* matrix){
	if(matrix)
		free(matrix);
}

void HelperOutputDim(int in_row, int in_col, int stride, int kernel_size, int &out_row, int &out_col, int padding_policy){
	// padding_policy = 0: same -> do padding on input so that the ouput size is the same as original input size; 
	// thus padding_size (p) = (k-1)/2
	// padding_policy = 1: valid -> no padding, output dim = ceil((n - k + 1)/s)
	
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
			cout << "padding_policy is not supported!" << endl; // to-do
			break;
	}
}

float* Convolution1D(float* matrix, int in_rows, int in_cols, int stride, int kernel_size, int padding_policy){
	int out_rows = 0;
	int out_cols = 0;
	HelperOutputDim(in_rows, in_cols, stride, kernel_size, out_rows, out_cols, padding_policy);

	float *output = AllocateTempBuffer(out_rows, out_cols);
	// policy = 0: same
	// policy = 1: valid
	int valid_offset = (padding_policy == 1) ? kernel_size / 2 : 0;
	
	for(int out_row_index = 0; out_row_index < out_rows; out_row_index++){
		for(int out_col_index = 0; out_col_index < out_cols; out_col_index++){
			float sum = 0;
			int offset_row = out_row_index * stride;
			int offset_col = out_col_index * stride;
			for(int k_row_index = -kernel_size/2; k_row_index <= kernel_size/2; k_row_index++){
				for(int k_col_index = -kernel_size/2; k_col_index <= kernel_size/2; k_col_index++){
					int in_row_index =  offset_row + k_row_index + valid_offset;
					int in_col_index =  offset_col + k_col_index + valid_offset;
					if(in_row_index < 0 || in_row_index >= in_rows || in_col_index < 0 || in_col_index >= in_cols)
						continue; // only happens for same mode, skipping is equivalent to summing with padded zeros
					sum += matrix[in_row_index * in_cols + in_col_index];
				}
			}
			output[out_row_index * out_cols + out_col_index] = sum;
		}
	}

	for(int i = 0; i < out_rows; i++){
		for(int j = 0; j < out_cols; j++){
			cout << output[i*out_cols + j] << " ";
		}
		cout << endl;
	}

	return output;
}

int main(){
	int n_rows = 5;
	int n_cols = 5;
	int stride = 1;
	int kernel_size = 3;
	int padding_policy = 0;
	
	
	float *input = AllocateTempBuffer(n_rows, n_cols);
	// test with all input element = 1
	for(int i = 0; i < n_rows; i++){
		for(int j = 0; j < n_cols; j++){
			input[i * n_cols + j] = 1;
		}
	}

	float *output = Convolution1D(input, n_rows, n_cols, stride, kernel_size, padding_policy);
	
	DeAllocateTempBuffer(input);
	DeAllocateTempBuffer(output);

	return 0;
}
