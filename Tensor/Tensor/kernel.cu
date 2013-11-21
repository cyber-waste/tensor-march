#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void factorAKernel ( int *T_i, float *T_d ,float *A, float *B, float *C, float *A_n, int l_i, int l_t, int l_q, int l_d, int k)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, error = 0.0f;
	
	int start = T_i[i];
	int end = ((i+1) < l_i) ? T_i[i+1] : l_d;
	int q = 0, t=0;
	
	for(int ind=start+2; ind<end; ind+=3){
		t = T_d[ind-2];
		q = T_d[ind-1];

		error = 0.0f;
		for(int j_i=0;j_i<k;j_i++){
			error += A[i*k+j_i]*B[t*k+j_i]*C[q*k+j_i];
		}

		temp = B[t*k+j]*C[q*k+j];
		sum_n += temp * T_d[ind] / error;
		sum_d += temp;
	}
	A_n[i*k+j] = A[i*k+j]*(sum_n/sum_d);
}

__global__ void factorBKernel ( int *T_t, float *T_d, float *A, float *B, float *C, float *B_n, int l_i, int l_t, int l_q, int l_d, int k){
	int t = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, error = 0.0f;
	
	int start = T_t[t];
	int end = ((t+1) < l_t) ? T_t[t+1] : l_d;
	int i,q;

	for(int ind=start+2; ind<end; ind+=3){
		q = (int)T_d[ind-2];
		i = (int)T_d[ind-1];

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

__global__ void factorCKernel ( int *T_q, float *T_d, float *A, float *B, float *C, float *C_n, int l_i, int l_t, int l_q, int l_d, int k){
	int q = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum_n = 0.1f, sum_d = 0.1f;
	float temp = 0.0f, error = 0.0f;
	
	int start = T_q[q];
	int end = ((q+1) < l_q) ? T_q[q+1] : l_d;
	int i,t;

	for(int ind=start+2; ind<end; ind+=3){
		i = (int)T_d[ind-2];
		t = (int)T_d[ind-1];

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

typedef struct tensor_elem{
    int i,t,q;
    float val;
} tensor_elem;

int comp_i(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.i > second.i || (first.i == second.i && first.t > second.t) || (first.i == second.i && first.t == second.t && first.q > second.q);
}

int comp_t(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.t > second.t || (first.t == second.t && first.q < second.q) || (first.t == second.t && first.q == second.q && first.i > second.i);
}

int comp_q(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.q > second.q || (first.q == second.q && first.i > second.i) || (first.q == second.q && first.i == second.i && first.t > second.t);
}

void parseTensorFile(char* fileName, int** Ti_ind, float** Ti_data, int** Tt_ind, float** Tt_data, int** Tq_ind, float** Tq_data, int* leni_ind, int* lent_ind, int* lenq_ind, int* num_values){
    
    FILE* f = fopen(fileName, "r");
    
    int n = 0;
    int i,t,q;
    float val;
    fscanf(f,"%d %d %d",&i,&t,&q);
    *leni_ind = i;
    *lent_ind = t;
    *lenq_ind = q;
    fscanf(f,"%d",&n);
    *num_values = n;
    tensor_elem* T = (tensor_elem*)malloc(n*sizeof(tensor_elem));
    
    for(int ind=0;ind<n;ind++){
        fscanf(f,"%d %d %d %f",&i,&t,&q,&val);
        tensor_elem cur;
        cur.i = i; cur.t = t; cur.q = q; cur.val = val;
        T[ind] = cur;
    }
    fclose(f);
    
    qsort(T, n, sizeof(tensor_elem),comp_i);
    
    *Ti_ind = (int*)malloc((*leni_ind)*sizeof(int));
    int* Ti_ind_cur = *Ti_ind;
    *Ti_data = (float*)malloc(3*n*sizeof(float));
    float* Ti_data_cur = *Ti_data; 
    
    int ind_data = 0;
    int ind_sparse = 0;
    Ti_ind_cur[0]=0;
    for(int ind=0;ind<((*leni_ind) - 1);ind++){
        while((ind_sparse < (3*n)) && (T[ind_data].i == ind)){
            Ti_data_cur[ind_sparse] = T[ind_data].t;
            Ti_data_cur[ind_sparse+1] = T[ind_data].q;
            Ti_data_cur[ind_sparse+2] = T[ind_data].val; 
            ind_data++;
            ind_sparse+=3;
        }
        Ti_ind_cur[ind+1]=ind_sparse;
                    
    }
    while(ind_sparse < (3*n)){
        Ti_data_cur[ind_sparse] = T[ind_data].t;
        Ti_data_cur[ind_sparse+1] = T[ind_data].q;
        Ti_data_cur[ind_sparse+2] = T[ind_data].val; 
        ind_data++;
        ind_sparse+=3;
    }
    
    qsort(T, n, sizeof(tensor_elem),comp_t);
    
	*Tt_ind = (int*)malloc((*lent_ind)*sizeof(int));
    int* Tt_ind_cur = *Tt_ind;
    *Tt_data = (float*)malloc(3*n*sizeof(float));
    float* Tt_data_cur = *Tt_data; 
    
    ind_data = 0;
    ind_sparse = 0;
    Tt_ind_cur[0]=0;
    for(int ind=0;ind<((*lent_ind) - 1);ind++){
        while((ind_sparse < (3*n)) && (T[ind_data].t == ind)){
            Tt_data_cur[ind_sparse] = T[ind_data].i;
            Tt_data_cur[ind_sparse+1] = T[ind_data].q;
            Tt_data_cur[ind_sparse+2] = T[ind_data].val; 
            ind_data++;
            ind_sparse+=3;
        }
        Tt_ind_cur[ind+1]=ind_sparse;
                    
    }
    while(ind_sparse < (3*n)){
        Tt_data_cur[ind_sparse] = T[ind_data].i;
        Tt_data_cur[ind_sparse+1] = T[ind_data].q;
        Tt_data_cur[ind_sparse+2] = T[ind_data].val; 
        ind_data++;
        ind_sparse+=3;
    }

    qsort(T, n, sizeof(tensor_elem),comp_q);
    
	*Tq_ind = (int*)malloc((*lenq_ind)*sizeof(int));
    int* Tq_ind_cur = *Tq_ind;
    *Tq_data = (float*)malloc(3*n*sizeof(float));
    float* Tq_data_cur = *Tq_data; 
    
    ind_data = 0;
    ind_sparse = 0;
    Tq_ind_cur[0]=0;
    for(int ind=0;ind<((*lenq_ind) - 1);ind++){
        while((ind_sparse < (3*n)) && (T[ind_data].q == ind)){
            Tq_data_cur[ind_sparse] = T[ind_data].i;
            Tq_data_cur[ind_sparse+1] = T[ind_data].t;
            Tq_data_cur[ind_sparse+2] = T[ind_data].val; 
            ind_data++;
            ind_sparse+=3;
        }
        Tq_ind_cur[ind+1]=ind_sparse;
                    
    }
    while(ind_sparse < (3*n)){
        Tq_data_cur[ind_sparse] = T[ind_data].i;
        Tq_data_cur[ind_sparse+1] = T[ind_data].t;
        Tq_data_cur[ind_sparse+2] = T[ind_data].val; 
        ind_data++;
        ind_sparse+=3;
    }

    free(T);
}

void printToFile(char* fileNameA, char* fileNameB, char* fileNameC, int k, float* A, int i, float* B, int t, float* C, int q){
    //open file
    FILE* fA = fopen(fileNameA, "w");
    FILE* fB = fopen(fileNameB, "w");
    FILE* fC = fopen(fileNameC, "w");
    for(int ind=1;ind<=(k*i);ind++){
        fprintf(fA,"%f",A[ind-1]);
        if(ind % k == 0) fprintf(fA,"\n");
        else fprintf(fA," ");
    }
    for(int ind=1;ind<=(k*t);ind++){
        fprintf(fB,"%f",B[ind-1]);
        if(ind % k == 0) fprintf(fB,"\n");
        else fprintf(fB," ");
    }
    for(int ind=1;ind<=(k*q);ind++){
        fprintf(fC,"%f",C[ind-1]);
        if(ind % k == 0) fprintf(fC,"\n");
        else fprintf(fC," ");
    }
    fclose(fA);
    fclose(fB);
    fclose(fC);
}

int main ( int argc, char *  argv [] )
{
	char fileName[] = "tensor.txt";
	char fileA[] = "A.txt";
	char fileB[] = "B.txt";
	char fileC[] = "C.txt";

    int *Ti_ind, *Tt_ind, *Tq_ind;
    float* Ti_data, *Tt_data, *Tq_data;
    int i, t, q, n;
    parseTensorFile(fileName,&Ti_ind,&Ti_data,&Tt_ind,&Tt_data,&Tq_ind,&Tq_data,&i,&t,&q,&n);
    /*
	for(int ind=0;ind<i;ind++){
        printf("%d ", Ti_ind[ind]);
    }
    printf("\n\n");
    */
	/*
	for(int ind=0;ind<(3*n);ind+=3){
        printf("%f %f %f\n",Ti_data[ind],Ti_data[ind+1],Ti_data[ind+2]);
    }
	*/
    printf("\n");   
	
	int k = 2;
	//scanf("%d", &k);
	
	float* A = new float[k*i];
	float* B = new float[k*t];
	float* C = new float[k*q];

	for(int ind=0;ind<(k*i);ind++) A[ind] = 1.0f;//(float)rand();
	for(int ind=0;ind<(k*t);ind++) B[ind] = 1.0f;//(float)rand();
	for(int ind=0;ind<(k*q);ind++) C[ind] = 1.0f;//(float)rand();

    float *Ti_data_cuda = NULL, *Tt_data_cuda = NULL, *Tq_data_cuda = NULL;
	int *Ti_ind_cuda = NULL, *Tt_ind_cuda = NULL, *Tq_ind_cuda = NULL;
	cudaMalloc ( (void**)&Ti_data_cuda, 3*n*sizeof(float) );
	cudaMalloc ( (void**)&Tt_data_cuda, 3*n*sizeof(float) );
	cudaMalloc ( (void**)&Tq_data_cuda, 3*n*sizeof(float) );
	cudaMalloc ( (void**)&Ti_ind_cuda, i*sizeof(int) );
	cudaMalloc ( (void**)&Tt_ind_cuda, t*sizeof(int) );
	cudaMalloc ( (void**)&Tq_ind_cuda, q*sizeof(int) );
	
	float *A_cuda = NULL, *B_cuda = NULL, *C_cuda = NULL, *A_next_cuda = NULL, *B_next_cuda = NULL, *C_next_cuda = NULL;
	int numBytesA = (k*i)*sizeof(float);
	int numBytesB = (k*t)*sizeof(float);
	int numBytesC = (k*q)*sizeof(float);
	cudaMalloc ( (void**)&A_cuda, numBytesA );
	cudaMalloc ( (void**)&B_cuda, numBytesB );
	cudaMalloc ( (void**)&C_cuda, numBytesC );
	cudaMalloc ( (void**)&A_next_cuda, numBytesA );
	cudaMalloc ( (void**)&B_next_cuda, numBytesB );
	cudaMalloc ( (void**)&C_next_cuda, numBytesC );

    dim3 blocks  = dim3(1, 1);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );
    cudaEventRecord ( start, 0 );
    
	cudaDeviceSynchronize();
	cudaMemcpy      ( Ti_data_cuda, Ti_data, 3*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy      ( Ti_ind_cuda, Ti_ind, i*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy      ( Tt_data_cuda, Tt_data, 3*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy      ( Tt_ind_cuda, Tt_ind, t*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy      ( Tq_data_cuda, Tq_data, 3*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy      ( Tq_ind_cuda, Tq_ind, q*sizeof(int), cudaMemcpyHostToDevice );
	
	cudaMemcpy      ( A_cuda, A, numBytesA, cudaMemcpyHostToDevice );
	cudaMemcpy      ( B_cuda, B, numBytesB, cudaMemcpyHostToDevice );
	cudaMemcpy      ( C_cuda, C, numBytesC, cudaMemcpyHostToDevice );

	
	bool flag = true;

	for(int ind=0;ind<1000;ind++){
		if(flag){	
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, dim3(k,i)>>>(Ti_ind_cuda, Ti_data_cuda, A_cuda, B_cuda, C_cuda, A_next_cuda,i,t,q,3*n,k);

			cudaDeviceSynchronize();
			factorBKernel<<<blocks, dim3(k,t)>>>(Tt_ind_cuda, Tt_data_cuda, A_next_cuda, B_cuda, C_cuda, B_next_cuda, i,t,q,3*n,k);

			cudaDeviceSynchronize();
			factorCKernel<<<blocks, dim3(k,q)>>>(Tq_ind_cuda, Tq_data_cuda, A_next_cuda, B_next_cuda,C_cuda,C_next_cuda,i,t,q,3*n,k);

		}
		else{
			cudaDeviceSynchronize();
			factorAKernel<<<blocks, dim3(k,i)>>>(Ti_ind_cuda, Ti_data_cuda, A_next_cuda, B_next_cuda, C_next_cuda, A_cuda, i,t,q,3*n,k);
			
			cudaDeviceSynchronize();
			factorBKernel<<<blocks, dim3(k,t)>>>(Tt_ind_cuda, Tt_data_cuda, A_cuda, B_next_cuda, C_next_cuda, B_cuda, i,t,q,3*n,k);
			
			cudaDeviceSynchronize();
			factorCKernel<<<blocks, dim3(k,q)>>>(Tq_ind_cuda, Tq_data_cuda, A_cuda, B_cuda, C_next_cuda, C_cuda, i,t,q,3*n,k);
		}
		flag = !flag;
    
	}
	
	cudaDeviceSynchronize();
	if(flag == false){
		cudaMemcpy      ( A, A_next_cuda, numBytesA, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_next_cuda, numBytesB, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_next_cuda, numBytesC, cudaMemcpyDeviceToHost );
	}
	else{
		cudaMemcpy      ( A, A_cuda, numBytesA, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( B, B_cuda, numBytesB, cudaMemcpyDeviceToHost );
		cudaMemcpy      ( C, C_cuda, numBytesC, cudaMemcpyDeviceToHost );
	}

	
	cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );
    cudaEventElapsedTime ( &gpuTime, start, stop );

    printf("\ntime spent executing by the GPU: %.2f millseconds\n", gpuTime );
    
	printToFile(fileA,fileB,fileC,k,A,i,B,t,C,q);

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop  );
    cudaFree(Ti_ind_cuda);
	cudaFree(Ti_data_cuda);
	cudaFree(Tt_ind_cuda);
	cudaFree(Tt_data_cuda);
	cudaFree(Tq_ind_cuda);
	cudaFree(Tq_data_cuda);
	cudaFree(A_cuda);
	cudaFree(B_cuda);
	cudaFree(C_cuda);
	cudaFree(A_next_cuda);
	cudaFree(B_next_cuda);
	cudaFree(C_next_cuda);
    delete[] Ti_ind;
	delete[] Ti_data;
	delete[] Tt_ind;
	delete[] Tt_data;
	delete[] Tq_ind;
	delete[] Tq_data;
	delete[] A;
	delete[] B;
	delete[] C;
	
    return 0;
}