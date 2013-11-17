#include <stdlib.h>
#include <stdio.h>

typedef struct tensor_elem{
    int i,t,q;
    float val;
} tensor_elem;

int comp_i(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.i < second.i || (first.i == second.i && first.t < second.t) || (first.i == second.i && first.t == second.t && first.q < second.q);
}

int comp_t(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.t < second.t || (first.t == second.t && first.q < second.q) || (first.t == second.t && first.q == second.q && first.i < second.i);
}

int comp_q(const void* el1, const void* el2){
    tensor_elem first  = *((tensor_elem*)el1);
    tensor_elem second = *((tensor_elem*)el2);
    return first.q < second.q || (first.q == second.q && first.i < second.i) || (first.q == second.q && first.i == second.i && first.t < second.t);
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
    
    qsort(T, n, sizeof(tensor_elem),comp_q);
    
    free(T);
}

void printToFile(char* fileNameA, char* fileNameB, char* fileNameC, int k, float* A, int i, float* B, int t, float* C, int q){
    //open file
    FILE* fA = fopen(fileNameA, "w");
    FILE* fB = fopen(fileNameB, "w");
    FILE* fC = fopen(fileNameC, "w");
    for(int ind=1;ind<=i;ind++){
        fprintf(fA,"%f",A[ind-1]);
        if(ind % k == 0) fprintf(fA,"\n");
        else fprintf(fA," ");
    }
    for(int ind=1;ind<=t;ind++){
        fprintf(fB,"%f",B[ind-1]);
        if(ind % k == 0) fprintf(fB,"\n");
        else fprintf(fB," ");
    }
    for(int ind=1;ind<=q;ind++){
        fprintf(fB,"%f",C[ind-1]);
        if(ind % k == 0) fprintf(fB,"\n");
        else fprintf(fB," ");
    }
    fclose(fA);
    fclose(fB);
    fclose(fC);
}

int main(){
    char fileName[] = "tensor.txt";
    int *Ti_ind;
    float* Ti_data;
    int i, t, q, n;
    parseTensorFile(fileName,&Ti_ind,&Ti_data,NULL,NULL,NULL,NULL,&i,&t,&q,&n);
    for(int ind=0;ind<i;ind++){
        printf("%d ", Ti_ind[ind]);
    }
    printf("\n\n");
    for(int ind=0;ind<(3*n);ind+=3){
        printf("%f %f %f\n",Ti_data[ind],Ti_data[ind+1],Ti_data[ind+2]);
    }
    printf("\n");   
    return 0;
}
