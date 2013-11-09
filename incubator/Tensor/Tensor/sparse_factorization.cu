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

int main ( int argc, char *  argv [] )
{
	int n,k;
	 
	// Tensor loading
	scanf("%d",&n);
	scanf("%d",&k);
	int size = n*n*n;
    int numBytesT = size * sizeof ( float );
	int numBytesABC = (n*k) * sizeof(float);

	
	float * A = new float [n*k];
	float * B = new float [n*k];
	float * C = new float [n*k];
	
	for(int i=0;i<(n*k);i++){
		A[i] = (float)(rand()%10000) + 1.0f;
		B[i] = (float)(rand()%10000) + 1.0f;
		C[i] = (float)(rand()%10000) + 1.0f;
	}

	float * T = new float [size];
	float *Q = new float[size];
	
    for ( int i = 0; i < size; i++ ){
        T[i] = (float)(rand()%10000) +1.0f;
		Q[i] = T[i];
	}


    float *T_c = NULL;
	cudaMalloc ( (void**)&T_c, numBytesT );
	
	float *A_cuda = NULL, *B_cuda = NULL, *C_cuda = NULL, *A_next_cuda = NULL, *B_next_cuda = NULL, *C_next_cuda = NULL;
	cudaMalloc ( (void**)&A_cuda, numBytesABC );
	cudaMalloc ( (void**)&B_cuda, numBytesABC );
	cudaMalloc ( (void**)&C_cuda, numBytesABC );
	cudaMalloc ( (void**)&A_next_cuda, numBytesABC );
	cudaMalloc ( (void**)&B_next_cuda, numBytesABC );
	cudaMalloc ( (void**)&C_next_cuda, numBytesABC );

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