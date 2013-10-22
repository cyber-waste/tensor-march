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
    
    def updateA():
        res = A
        
        for i in range(len(A)) :
            for j in range(len(A[0])) :
                k1, k2 = 0.1, 0.1
                for t in range(len(B)) :
                    for q in range(len(C)) :
                        k1 += B[t][j]*C[q][j]*tensor[q][i][t]/error[q][i][t]
                        k2 += B[t][j]*C[q][j]
                res[i][j] = A[i][j]*k1/k2
        
        return res
    
    def updateB():
        res = B
        
        for t in range(len(B)) :
            for j in range(len(A[0])) :
                k1, k2 = 0.1, 0.1
                for i in range(len(A)) :
                    for q in range(len(C)) :
                        k1 += A[i][j]*C[q][j]*tensor[q][i][t]/error[q][i][t]
                        k2 += A[i][j]*C[q][j]
                res[t][j] = B[t][j]*k1/k2
        
        return res
    
    def updateC():
        res = C
        
        for q in range(len(C)) :
            for j in range(len(A[0])) :
                k1, k2 = 0.1, 0.1
                for i in range(len(A)) :
                    for t in range(len(B)) :
                        k1 += A[i][j]*B[t][j]*tensor[q][i][t]/error[q][i][t]
                        k2 += A[i][j]*B[t][j]
                res[q][j] = C[q][j]*k1/k2
        
        return res
    
    i = len(tensor[0])
    q = len(tensor[0][0])
    t = len(tensor)
    A = gen_matrix(i, j)
    B = gen_matrix(q, j)
    C = gen_matrix(t, j)
    
    for _ in range(iter) :
        error = build_tensor(A, B, C)
        A = updateA()
        B = updateB()
        C = updateC()
    
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
(A_new,B_new,C_new) = factorize(tensor, 2, 10000)

print_tensor(build_tensor(A_new, B_new, C_new))