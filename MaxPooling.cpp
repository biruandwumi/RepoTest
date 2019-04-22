#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void HelperOutputDim(int in_row, int in_col, int stride, int kernel_size, int &out_row, int &out_col, int padding_policy){
	// padding_policy = 0: any
	// padding_policy = 1: valid
	switch(padding_policy){
		case 0: // any
			out_row = (int)ceil((float)in_row / (float)stride);
			out_col = (int)ceil((float)in_col / (float)stride);
			break;
		case 1: // valid
			out_row = (int)ceil((float)(in_row - kernel_size + 1) / (float)stride);
			out_col = (int)ceil((float)(in_col - kernel_size + 1) / (float)stride);
			break;
		default:
			cout<< "padding_policy is not supported!" << endl; // to-do
			break;
	}
}

float* AllocateTempBuffer(int n_rows, int n_cols){
	float *matrix = (float*)malloc(n_rows * n_cols * sizeof(float*));
	// initialization
	for(int i = 0; i < n_rows; i++){
		for(int j = 0; j < n_cols; j++){
			matrix[i * n_cols + j] = (float)(i * n_cols + j);
		}
	}
	return matrix;
}

void DeAllocateTempBuffer(float* matrix){
	if(matrix)
		free(matrix);
}

float* MaxPooling1D(float* matrix, int in_row, int in_col, int stride, int kernel_size, int padding_policy){
	int out_row = 0;
	int out_col = 0;
	HelperOutputDim(in_row, in_col, stride, kernel_size, out_row, out_col, padding_policy);

	float *output = AllocateTempBuffer(out_row, out_col);
	for(int out_row_index = 0; out_row_index < out_row; out_row_index++){
		for(int out_col_index = 0; out_col_index < out_col; out_col_index++){
			int offset_row = out_row_index*stride;
			int offset_col = out_col_index*stride;
			for(int k_row_index = offset_row; k_row_index < offset_row + kernel_size; k_row_index++){
				for(int k_col_index = offset_col; k_col_index < offset_col + kernel_size; k_col_index++){
					if(k_row_index >= in_row || k_col_index >= in_col)
						continue;
					output[out_row_index * out_col + out_col_index] = 
						max(output[out_row_index * out_col + out_col_index], matrix[k_row_index * in_col + k_col_index]);
				}
			}
		}
	}
	for(int i = 0; i < out_row; i++){
		for(int j = 0; j < out_col; j++){
			cout << output[i*out_col + j] << " ";
		}
		cout << endl;
	}
	return output;
}

int main(){
	int n_rows = 4;
	int n_cols = 3;
	int stride = 2;
	int kernel_size = 2;
	int padding_policy = 0;
	
	float *input = AllocateTempBuffer(n_rows, n_cols);
	float *output = MaxPooling1D(input, n_rows, n_cols, stride, kernel_size, padding_policy);
	
	DeAllocateTempBuffer(input);
	DeAllocateTempBuffer(output);

	return 0;

}
