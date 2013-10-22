'''
Created on Oct 22, 2013

@author: vlg
'''

import random

def gen_matrix(n,m):
    return [[ random.randint(0,1000) for _ in range(m)] for _ in range(n)]
    
def gen_tensor(n):
    return [ gen_matrix(n,n) for _ in range(n)]

def print_tensor(tensor):
    N = len(tensor)
    for i in range(N) :
        for j in range(N) :
            print tensor[i][j]
        print

def print_matrix(matrix):
    for i in range(len(matrix)) :
        print matrix[i]
        print

def factorize(tensor,j,iter=10):
    
    def update(A,B,C):
        res = A
        
        for i in range(len(A)) :
            for j in range(len(A[0])) :
                k1, k2 = 0.0, 0.0
                for t in range(len(B)) :
                    for q in range(len(C)) :
                        k1 += B[t][j]*C[q][j]*tensor[t][i][q]/error[t][i][q]
                        k2 += B[t][j]*C[q][j]
                res[i][j] = A[i][j]*k1/k2
        
        return res
    
    i = len(tensor[0])
    q = len(tensor[0][0])
    t = len(tensor)
    A = gen_matrix(i, j)
    B = gen_matrix(q, j)
    C = gen_matrix(t, j)
    
    for _ in range(iter) :
        error = build_tensor(A, B, C)
        A = update(A,B,C)
        B = update(B, A, C)
        C = update(C,A,B)
    
    return (A,B,C)

def build_tensor(A,B,C):
    i = len(A)
    q = len(B)
    t = len(C)
    j = len(A[0])
    
    res = [ [ [ sum([ A[ii][ji]*B[qi][ji]*C[ti][ji] for ji in range(j)]) for qi in range(q)] for ii in range(i)] for ti in range(t)]
    return res
    
A = [[1,2],[3,4]]
B = [[5,6],[7,8],[9,10]]
C = [[11,12],[13,14]]

tensor = build_tensor(A, B, C)
print_tensor(tensor)
# (A_new,B_new,C_new) = factorize(tensor, 2)
# 
# print_matrix(A_new)
# print_matrix(B_new)
# print_matrix(C_new)