#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__device__ float getTensorElement(int *T_i, float *T_d, int l_i, int l_d, int i, int t, int q){
	int start = T_i[i];
	int end = ((i+1) < l_i) ? T_i[i+1] : l_d;
	float res = 0.0f;
	for(int i=start+2;i<end;i+=3){
		if(T_d[i-2] > t) break;
		else if(T_d[i-2] == t && T_d[i-1] == q){
			res = T_d[i];
			break;
		}
	}
	return res;
}

__global__ void factorAKernel ( float *T_i, float *T_d, int l_i, int l_d ,float *A, float *B, float *C, float *A_n, int n, int k)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, ratio = 0.0f, error = 0.0f;
	
	int start = T_i[i];
	int end = ((i+1) < l_i) ? T_i[i+1] : l_d;
	int t,q;

	for(int ind=start+2; ind<end; ind+=3){
		t = T_d[ind-2];
		q = T_d[ind-1];

		error = 0.0f;
		for(int j=0;j<k;j++){
			error += A[i*k+j]*B[t*k+j]*C[q*k+j];
		}

		temp = B[t*k+j]*C[q*k+j];
		sum_n += temp * T_d[ind] / error;
		sum_d += temp;
	}
	
	A_n[i*k+j] = A[i*k+j]*(sum_n/sum_d);
}

__global__ void factorBKernel ( float *T_t, float *T_d, int l_t, int l_d, float *A, float *B, float *C, float *B_n, int n, int k){
	int t = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, ratio = 0.0f, error = 0.0f;
	
	int start = T_t[t];
	int end = ((t+1) < l_t) ? T_t[t+1] : l_d;
	int i,q;

	for(int ind=start+2; ind<end; ind+=3){
		q = T_d[ind-2];
		i = T_d[ind-1];

		error = 0.0f;
		for(int j=0;j<k;j++){
			error += A[i*k+j]*B[t*k+j]*C[q*k+j];
		}

		temp = A[i*k+j]*C[q*k+j];
		sum_n += temp * T_d[ind] / error;
		sum_d += temp;
	}

	B_n[t*k+j] = B[t*k+j]*(sum_n/sum_d);
}

__global__ void factorCKernel ( float *T_q, float *T_d, int l_q, int l_d, float *A, float *B, float *C, float *C_n, int n, int k){
	int q = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, ratio = 0.0f, error = 0.0f;
	
	int start = T_q[q];
	int end = ((q+1) < l_q) ? T_q[q+1] : l_d;
	int i,t;

	for(int ind=start+2; ind<end; ind+=3){
		i = T_d[ind-2];
		t = T_d[ind-1];

		error = 0.0f;
		for(int j=0;j<k;j++){
			error += A[i*k+j]*B[t*k+j]*C[q*k+j];
		}

		temp = A[i*k+j]*B[t*k+j];
		sum_n += temp * T_d[ind] / error;
		sum_d += temp;
	}

	C_n[q*k+j] = C[q*k+j]*sum_n/sum_d;
}

void parseTensorFile(char* fileName, int** Ti_ind, float** Ti_data, int* leni_ind, int* leni_data, int** Tt_ind, float** Tt_data, int* lent_ind, int* lent_data, int** Tq_ind, float** Tq_data, int* lenq_ind, int* lenq_data){

}

int main ( int argc, char *  argv [] )
{
	char fileName[256];
	int dim_i, dim_t, dim_q;
	int k;

	scanf("%s",fileName);
	scanf("%s", &k);
	
	float *Ti_data, *Tt_data, Tq_data;
	int *Ti_ind, *Tt_ind, *Tq_ind;
	int leni_ind, leni_data, lent_ind, lent_data, lenq_ind, lenq_data;
	parseTensorFile(fileName, &Ti_ind, &Ti_data, &leni_ind, &leni_data, &Tt_ind, &Tt_data, &lent_ind, &lent_data, &Tq_ind, &Tq_data, &lenq_ind, &lenq_data);
	dim_i = leni_ind;
	dim_t = lent_ind;
	dim_q = lenq_ind;

	float* A = new float[k*dim_i];
	float* B = new float[k*dim_t];
	float* C = new float[k*dim_q];

	for(int i=0;i<(k*dim_i);i++) A[i] = (float)rand();
	for(int i=0;i<(k*dim_t);i++) B[i] = (float)rand();
	for(int i=0;i<(k*dim_q);i++) C[i] = (float)rand();

    float *Ti_data_cuda = NULL, *Tt_data_cuda = NULL, Tq_data_cuda = NULL;
	int *Ti_ind_cuda = NULL, *Tt_ind_cuda = NULL, *Tq_ind_cuda = NULL;
	cudaMalloc ( (void**)&Ti_data_cuda, leni_data*sizeof(float) );
	cudaMalloc ( (void**)&Tt_data_cuda, lent_data*sizeof(float) );
	cudaMalloc ( (void**)&Tq_data_cuda, lenq_data*sizeof(float) );
	cudaMalloc ( (void**)&Ti_ind_cuda, leni_ind*sizeof(int) );
	cudaMalloc ( (void**)&Tt_ind_cuda, lent_ind*sizeof(int) );
	cudaMalloc ( (void**)&Tq_ind_cuda, lenq_ind*sizeof(int) );
	
	float *A_cuda = NULL, *B_cuda = NULL, *C_cuda = NULL, *A_next_cuda = NULL, *B_next_cuda = NULL, *C_next_cuda = NULL;
	int numBytesA = (k*dim_i)*sizeof(float);
	int numBytesB = (k*dim_t)*sizeof(float);
	int numBytesC = (k*dim_q)*sizeof(float);
	cudaMalloc ( (void**)&A_cuda, numBytesA );
	cudaMalloc ( (void**)&B_cuda, numBytesB );
	cudaMalloc ( (void**)&C_cuda, numBytesC );
	cudaMalloc ( (void**)&A_next_cuda, numBytesA );
	cudaMalloc ( (void**)&B_next_cuda, numBytesB );
	cudaMalloc ( (void**)&C_next_cuda, numBytesC );

    dim3 threads = dim3(k, n);
    dim3 blocks  = dim3(1, 1);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    cudaEventRecord ( start, 0 );
    
	cudaDeviceSynchronize();
	cudaMemcpy      ( T_c, T, numBytesT, cudaMemcpyHostToDevice );
	cudaMemcpy      ( Q_c, Q, numBytesT, cudaMemcpyHostToDevice );
	
	cudaMemcpy      ( A_cuda, A, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( B_cuda, B, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( C_cuda, C, numBytesABC, cudaMemcpyHostToDevice );

	
	bool flag = true;

	int l_i, l_d;

	for(int i=0;i<1000;i++){
		if(flag){	
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, threads>>>(Ti_ind_cuda, Ti_data_cuda, l_i, l_d_i, A_cuda, B_cuda, C_cuda, A_next_cuda,n,k);

			cudaDeviceSynchronize();
			factorBKernel<<<blocks, threads>>>(Tt_ind_cuda, Tt_data_cuda, l_t, l_d_t, A_next_cuda, B_cuda, C_cuda, B_next_cuda, n,k);

			cudaDeviceSynchronize();
			factorCKernel<<<blocks, threads>>>(Tq_ind_cuda, Tq_data_cuda, l_q, l_d_q, A_next_cuda, B_next_cuda,C_cuda,C_next_cuda,n,k);

		}
		else{
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, threads>>>(T_c,Q_c,A_next_cuda,B_next_cuda,C_next_cuda,A_cuda,n,k);
			
			cudaDeviceSynchronize();
			factorBKernel<<<blocks, threads>>>(T_c,Q_c,A_cuda,B_next_cuda,C_next_cuda,B_cuda,n,k);
			
			cudaDeviceSynchronize();
			factorCKernel<<<blocks, threads>>>(T_c,Q_c,A_cuda,B_cuda,C_next_cuda,C_cuda,n,k);
		}
		flag = !flag;
    
	}
	
	cudaDeviceSynchronize();
	if(flag == false){
		cudaMemcpy      ( A, A_next_cuda, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_next_cuda, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_next_cuda, numBytesABC, cudaMemcpyDeviceToHost );
	}
	else{
		cudaMemcpy      ( A, A_cuda, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_cuda, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_cuda, numBytesABC, cudaMemcpyDeviceToHost );
	}
	cudaMemcpy      ( Q, Q_c, numBytesT, cudaMemcpyDeviceToHost );

	
	cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );
    cudaEventElapsedTime ( &gpuTime, start, stop );

    printf("\ntime spent executing by the GPU: %.2f millseconds\n", gpuTime );
    
	printf("Matrix A\n");
	//for ( int i = 0; i < (n*k); i++ ) printf ( "%f ", A[i] );
	printf("\n");

	printf("Matrix B\n");
	//for ( int i = 0; i < (n*k); i++ ) printf ( "%f ", B[i] );
	printf("\n");

	printf("Matrix C\n");
	//for ( int i = 0; i < (n*k); i++ ) printf ( "%f ", C[i] );
	printf("\n");
	
	printf("Tensor Q\n");
	//for(int i=0;i<(n*n*n);i++) printf("%f ", Q[i]);
	printf("\n");

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop  );
    cudaFree(T_c);
	cudaFree(Q_c);
	cudaFree(A_cuda);
	cudaFree(B_cuda);
	cudaFree(C_cuda);
	cudaFree(A_next_cuda);
	cudaFree(B_next_cuda);
	cudaFree(C_next_cuda);
    delete[] T;
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] Q;

    return 0;
}