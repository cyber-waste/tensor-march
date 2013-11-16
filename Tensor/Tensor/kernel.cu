
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void errorKernel(float* Q, float* A, float* B, float* C, int n, int k){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.z * blockDim.z + threadIdx.z;
	float sum = 0.0f;
	for(int j=0;j<k;j++){
		sum += A[i*k+j]*B[t*k+j]*C[q*k+j];
	}
	Q[n*n*q+n*i+t] = sum;
}

__global__ void factorAKernel ( float *T, float *Q, float *A, float *B, float *C, float *A_n, int n, int k)
{
	//printf("");
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f;
		
	for(int t=0; t<n; t++){
		for(int q=0; q<n; q++){
			temp = B[t*k+j]*C[q*k+j];
			/*
			if(i==0 && j==0){
				printf("(t,q)=(%d %d) num=%f den=%f\n",t,q,temp*(T[n*n*q+n*i+t]/ Q[n*n*q+n*i+t]),temp);
			}
			*/
			// ugly fix
			//if(Q[n*n*q+n*i+t] < 0.00000001)
			//	sum_n += temp;
			//else
			sum_n += temp*(T[n*n*q+n*i+t]/ Q[n*n*q+n*i+t]);
			sum_d += temp;
		}
	}
		
	A_n[i*k+j] = A[i*k+j]*(sum_n/sum_d);
}

__global__ void factorBKernel ( float *T, float *Q, float *A, float *B, float *C, float *B_n, int n, int k){
	int t = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f;
	
	for(int i=0;i<n;i++){
		for(int q=0;q<n;q++){
			temp = A[i*k+j]*C[q*k+j];
			/*
			if(t==0 && j==0){
				printf("(i,q)=(%d %d) num=%f den=%f\n",i,q,temp*(T[n*n*q+n*i+t]/ Q[n*n*q+n*i+t]),temp);
			}
			*/
			//if(Q[n*n*q+n*i+t] < 0.00000001)
			//	sum_n += temp;
			//else
			sum_n += temp*(T[n*n*q+n*i+t]/ Q[n*n*q+n*i+t]);
			sum_d += temp;
		}
	}
	//if(t==0 && j==0) printf("res = %f * %f / %f\n",B[t*k+j],sum_n,sum_d);
	B_n[t*k+j] = B[t*k+j]*(sum_n/sum_d);
}

__global__ void factorCKernel ( float *T, float *Q, float *A, float *B, float *C, float *C_n, int n, int k){
	int q = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f;

	for(int t=0;t<n;t++){
		for(int i=0;i<n;i++){
			temp = A[i*k+j]*B[t*k+j];
			//if(Q[n*n*q+n*i+t] < 0.00000001)
			//	sum_n += temp;
			//else
			sum_n += temp*T[n*n*q+n*i+t]/ Q[n*n*q+n*i+t];
			sum_d += temp;
		}
	}

	C_n[q*k+j] = C[q*k+j]*sum_n/sum_d;
}

float* buildTensor(float* A, float* B, float* C, int n, int k){
	int size = n*n*n;
	float* T = new float[size];

	for(int i=0;i<n;i++){
		for(int t=0;t<n;t++){
			for(int q=0;q<n;q++){
				T[n*n*q+n*i+t] = 0;
				for(int j=0;j<k;j++){
					T[n*n*q+n*i+t] += A[i*k+j]*B[t*k+j]*C[q*k+j];
				}
			}
		}
	}

	return T;
}

float* buildTensorExample(){
	float A[] = {1.0f,2.0f,3.0f,4.0f};
	float B[] = {5.0f,6.0f,7.0f,8.0f};
	float C[] = {9.0f,10.0f,11.0f,12.0f};
	int n=2, k=2;
	return buildTensor(A,B,C,n,k);
}

int main ( int argc, char *  argv [] )
{
    //int n=2, k=2;
	int n,k;
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
		//Q[i] = T[i];//(float)(rand()%100);
		//printf("%f ",T[i]);
	}
	
	/*
	float* A_n = new float[n*k];
	float* B_n = new float[n*k];
	float* C_n = new float[n*k];
	for(int i=0;i<(n*k);i++){
		A_n[i] = B_n[i] = C_n[i]= 0.0f;
	}
	*/
	//float  *Q = buildTensor(A,B,C,n,k);
	//float *T = buildTensorExample();

    float *T_c = NULL, *A_c = NULL, *B_c = NULL, *C_c = NULL, *Q_c = NULL, *A_n_c = NULL, *B_n_c = NULL, *C_n_c = NULL;
    cudaMalloc ( (void**)&T_c, numBytesT );
	cudaMalloc( (void**)&Q_c, numBytesT);
	cudaMalloc ( (void**)&A_c, numBytesABC );
	cudaMalloc ( (void**)&B_c, numBytesABC );
	cudaMalloc ( (void**)&C_c, numBytesABC );
	cudaMalloc ( (void**)&A_n_c, numBytesABC );
	cudaMalloc ( (void**)&B_n_c, numBytesABC );
	cudaMalloc ( (void**)&C_n_c, numBytesABC );

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
	
	cudaMemcpy      ( A_c, A, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( B_c, B, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( C_c, C, numBytesABC, cudaMemcpyHostToDevice );

	/*	
	cudaDeviceSynchronize();
	cudaMemcpy      ( A_n_c, A_n, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( B_n_c, B_n, numBytesABC, cudaMemcpyHostToDevice );
	cudaMemcpy      ( C_n_c, C_n, numBytesABC, cudaMemcpyHostToDevice );
	*/
	/*
	cudaDeviceSynchronize();
	factorAKernel<<<blocks, threads>>>(T_c,Q_c,A_c,B_c,C_c,A_n_c,n,k);
	cudaDeviceSynchronize();
	factorBKernel<<<blocks, threads>>>(T_c,Q_c,A_n_c,B_c,C_c,B_n_c,n,k);
	cudaDeviceSynchronize();
	factorCKernel<<<blocks, threads>>>(T_c,Q_c,A_n_c,B_n_c,C_c,C_n_c,n,k);
	cudaDeviceSynchronize();
	
	cudaMemcpy      ( A, A_n_c, numBytesABC, cudaMemcpyDeviceToHost );
	cudaMemcpy      ( B, B_n_c, numBytesABC, cudaMemcpyDeviceToHost );
	cudaMemcpy      ( C, C_n_c, numBytesABC, cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();
	*/
	
	bool flag = true;

	for(int i=0;i<1000;i++){
		if(flag){	
			cudaDeviceSynchronize();
			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_c,B_c,C_c,n,k);
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, threads>>>(T_c,Q_c,A_c,B_c,C_c,A_n_c,n,k);
			cudaDeviceSynchronize();

			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_n_c,B_c,C_c,n,k);
			cudaDeviceSynchronize();
			factorBKernel<<<blocks, threads>>>(T_c,Q_c,A_n_c,B_c,C_c,B_n_c,n,k);
			cudaDeviceSynchronize();
			
			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_n_c,B_n_c,C_c,n,k);
			cudaDeviceSynchronize();
			factorCKernel<<<blocks, threads>>>(T_c,Q_c,A_n_c,B_n_c,C_c,C_n_c,n,k);
			cudaDeviceSynchronize();

		}
		else{
			cudaDeviceSynchronize();
			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_n_c,B_n_c,C_n_c,n,k);
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, threads>>>(T_c,Q_c,A_n_c,B_n_c,C_n_c,A_c,n,k);
			cudaDeviceSynchronize();
			
			cudaDeviceSynchronize();
			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_c,B_n_c,C_n_c,n,k);
			factorBKernel<<<blocks, threads>>>(T_c,Q_c,A_c,B_n_c,C_n_c,B_c,n,k);
			cudaDeviceSynchronize();
			
			cudaDeviceSynchronize();
			errorKernel<<<dim3(1,1,1),dim3(n,n,n)>>>(Q_c,A_c,B_c,C_n_c,n,k);
			factorCKernel<<<blocks, threads>>>(T_c,Q_c,A_c,B_c,C_n_c,C_c,n,k);
			cudaDeviceSynchronize();
		}
		flag = !flag;
    
	}
	
	cudaDeviceSynchronize();
	if(flag == false){
		cudaMemcpy      ( A, A_n_c, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_n_c, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_n_c, numBytesABC, cudaMemcpyDeviceToHost );
	}
	else{
		cudaMemcpy      ( A, A_c, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_c, numBytesABC, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_c, numBytesABC, cudaMemcpyDeviceToHost );
	}
	cudaMemcpy      ( Q, Q_c, numBytesT, cudaMemcpyDeviceToHost );
	
	cudaDeviceSynchronize();
	
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
	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);
	cudaFree(A_n_c);
	cudaFree(B_n_c);
	cudaFree(C_n_c);
    delete[] T;
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] Q;

    return 0;
}