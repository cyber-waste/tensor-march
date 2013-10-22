'''
Created on Oct 22, 2013

@author: vlg
'''

import random

def gen_tensor(N):
    def gen_matrix():
        return [[ random.randint(0,1000) for _ in range(N)] for _ in range(N)]
    
    return [ gen_matrix() for _ in range(N)]

def print_tensor(tensor):
    N = len(tensor)
    for i in range(N) :
        for j in range(N) :
             print tensor[i][j]
        print

def factorize(tensor):
    pass

print_tensor(gen_tensor(2))