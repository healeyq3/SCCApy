import random
import math
from random import sample

import numpy as np
# from scipy import linalg as LA
import scipy

## Given dimensions (n1, m2, s1, s2), generate the synthetic data 
## and compute the upper bound at the root node 
def gen_data(n1, m2, s1, s2):
    global A
    global B, mu
    global C, nu
    
    np.random.seed(1)
    random.seed(1)
    
    B = np.matrix(np.random.normal(0, 1, (n1, n1)))
    B = B*B.T + np.matrix(np.eye(n1, dtype=float))

    C = np.matrix(np.random.normal(0, 1, (m2, m2)))
    C = C*C.T + np.matrix( np.eye(m2, dtype=float)) 
    
    NS1 = sample(list(range(n1)), n1-s1)
    NS2 = sample(list(range(m2)), m2-s2)
    
    u = np.matrix(np.random.uniform(0, 1, (n1)))
    for i in NS1:
        u[0,i] = 0   
    u = u/math.sqrt((u*B*u.T)[0,0])
    
    v = np.matrix(np.random.uniform(0, 1, (m2)))
    for i in NS2:
        v[0,i] = 0   
    v = v/math.sqrt((v*C*v.T)[0,0])
    
    l = np.random.uniform(0,1)

    A = l*B*u.T*v*C
    
    cov = np.block([[B, A], [A.T, C]])
    mean = [0.0]*(n1+m2)
    
    N = 5000
    np.random.seed(1)
    temp = np.random.multivariate_normal(mean, cov, N)
    X = np.matrix(temp[:,0:n1])
    Y = np.matrix(temp[:, n1:(n1+m2)])
    
    B = sum([X[i,:].T*X[i,:] for i in range(N)])/N
    
    C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N
    
    A = sum([X[i,:].T*Y[i,:] for i in range(N)])/N
    temp1 = np.matrix(scipy.linalg.sqrtm(B.I))
    temp2 = np.matrix(scipy.linalg.sqrtm(C.I))
    u, sigma, v = np.linalg.svd(temp1 * A * temp2) 
    print('the upper bound at the root node is ',  max(sigma)) ## the upper bound at the root node 
    return max(sigma)


## Greedy and local search algorithms are used to generate a feasible solution
def greedy(n1, m2, s1, s2):
    
    s = min(s1, s2)
    sn = list(range(n1))
    sm = list(range(m2))
    bestf = 0.0
    
    start = datetime.datetime.now()
    
    # initialize two subsets
    S1 = []
    S2 = []  
    temp = np.zeros([n1, m2])
    
    for i in range(n1):
        for j in range(m2):
            temp[i,j] = abs(1/math.sqrt(B[i,i]) * A[i,j] * 1/math.sqrt(C[j,j]))
    
    sindex = np.unravel_index(np.argmax(abs(temp), axis=None), temp.shape)
    
    S1.append(sindex[0])
    S2.append(sindex[1])    
    
    for i in range(s-1):
        unsel = []
        unsel = list(set(sn) - set(S1)) 
        bestLB = 0.0
        bestl = 0
   
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for l in unsel:
            S1.append(l)  
            temp1 = B[np.ix_(S1, S1)]
            temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
            
            u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
            
            LB = max(sigma)
            
            if LB > bestLB:
                bestLB = LB
                bestl = l
            S1.remove(l)
            
        S1.append(bestl)
        temp1 = B[np.ix_(S1, S1)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        
        # then update set S2    
        unsel = []
        unsel = list(set(sm) - set(S2)) 
        bestLB = 0.0
        bestl = 0
        for l in unsel:
            S2.append(l)
            temp2 = C[np.ix_(S2, S2)]
            temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
            u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 

            LB = 0.0
            LB =max(sigma)

            if LB > bestLB:
                bestLB = LB
                bestl = l
            S2.remove(l)
            
        S2.append(bestl)
        
    if s1 < s2:
        temp1 = B[np.ix_(S1, S1)]

        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        
        for i in range(s, s2):
            # only update set S2    
            unsel = []
            unsel = list(set(sm) - set(S2)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S2.append(l)
                temp2 = C[np.ix_(S2, S2)]
                temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 

                LB = 0.0
                LB =max(sigma)

                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S2.remove(l)
                
            S2.append(bestl)
    else:
        
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for i in range(s, s1):
            # only update set S1
            unsel = []
            unsel = list(set(sn) - set(S1)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S1.append(l)  
                temp1 = B[np.ix_(S1, S1)]
                temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                    
                S1.remove(l)
                
            S1.append(bestl)
            
    end = datetime.datetime.now()
    time = (end-start).seconds
    bestf = bestLB
    
    return  S1, S2, bestf, time  


def localsearch(n1, m2, s1, s2):
    start = datetime.datetime.now()
    S1, S2, bestf, time = greedy(n1, m2, s1, s2)
    
    sn = list(range(n1))
    sm = list(range(m2))
    
    optimal = False

    unsel = []
    unsel = list(set(sn) - set(S1)) 
    
    unsel2 = []
    unsel2 = list(set(sm) - set(S2))
    
    while(optimal == False):
        optimal = True
        
        # first update row selection
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for i in S1:
            for j in unsel:
                S1.remove(i)
                S1.append(j)
                
                temp1 = B[np.ix_(S1, S1)]
                temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestf:
                    print(i)
                    optimal = False                             
                    bestf = LB
                    unsel.append(i)
                    unsel.remove(j)
                    break
                
                S1.append(i)
                S1.remove(j)
                

        temp1 = B[np.ix_(S1, S1)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        for i in S2:
            for j in unsel2:
                S2.remove(i)
                S2.append(j)

                temp2 = C[np.ix_(S2, S2)]
                temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 


                if LB > bestf:
                    print(i)
                    optimal = False                             
                    bestf = LB
                    unsel2.append(i)
                    unsel2.remove(j)
                    break
                
                S2.append(i)
                S2.remove(j)
                
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    print('the lower bound at the root node is ',  bestf) 
    
    return bestf, time
    

### Variable fixing function ###
## Here, we only fix binary variables to be 1
## Output set S1 that contains variables being 1 for two groups
# Also output scores of (n1+m2) variables used for branching, e.g., branch a variable with the smallest score.
def varfix(n1, m2, LB):
    scores=[]
    OptS1 = []
    sn = list(range(n1))
    sm = list(range(m2))
    temp2 = np.matrix(scipy.linalg.sqrtm(C.I))
    for i in range(n1):
        sn.remove(i)
        temp1 = B[np.ix_(sn, sn)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(sn, sm)] * temp2)
        sn.append(i)
        scores.append(max(sigma))
        if max(sigma) < LB:
            # print('S1:', i, max(sigma), LB)
            OptS1.append(i)
            
    OptS2 = []
    sn = list(range(n1))
    sm = list(range(m2))
    temp1 = np.matrix(scipy.linalg.sqrtm(B.I))
    for i in range(m2):
        sm.remove(i)
        temp2 = C[np.ix_(sm, sm)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(sn, sm)] * temp2)
        sm.append(i)
        scores.append(max(sigma))
        if max(sigma) < LB:
            # print('S2:', i, max(sigma), LB)
            OptS2.append(i)
            
    S1 = []
    for i in OptS1:
        S1.append(i)
    for i in OptS2:
        S1.append(i+n1)
    print("the fixed variables being 1 at the root node are ", S1)  
    return S1, scores


### Compute upper bound at each node###
## Input: n1, m2, S1, S0
## S1: the number of total variables to be 1
## S0: the number of total variables to be 0
def bound(n1, m2, S1, S0):
    zeroS1, zeroS2 = [], []
    for i in S0:
        if i <= n1:
            zeroS1.append(i)
        else:
            zeroS2.append(i-n1)
            
    sn = list(range(n1))
    sm = list(range(m2))
        
    sel1 = []
    sel1 = list(set(sn) - set(zeroS1)) 
    
    sel2 = []
    sel2 = list(set(sm) - set(zeroS2))
    
    temp1 = B[np.ix_(sel1, sel1)]
    temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
    temp2 = C[np.ix_(sel2, sel2)]
    temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
    u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(sel1, sel2)] * temp2)
    
    return max(sigma)   
     

### Compute objective at the terminal node###
## Input: n1, m2, S1, S0
## Output: objective value
def fval(n1, m2, S1, S0):
    oneS1, oneS2 = [], []
    for i in S1:
        if i < n1:
            oneS1.append(i)
        else:
            oneS2.append(i-n1)
    
    temp1 = B[np.ix_(oneS1, oneS1)]
    temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))

    temp2 = C[np.ix_(oneS2, oneS2)]
    temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
    u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(oneS1, oneS2)] * temp2) 
    return max(sigma) 

if __name__ == "__main__":
    # n1, m2, s1, s2 = 40, 40, 10, 10
    n1, m2, s1, s2 = 20, 20, 10, 10
    n = n1+m2

    ## generate data and compute upper bound at root node
    UB = gen_data(n1, m2, s1, s2)
    print(type(B))
    # S1 = [1, 19, 24, 34]
    # S0 = [0, 15, 39]
    S1 = []
    S0 = []
    new_UB = bound(n1, m2, S1, S0)
    print("new UB:", new_UB)

    # S1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    S1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    print(len(S1))

    val = fval(n1, m2, S1=S1, S0=[])
    print(val)

    ## compute lower bound at root node
    # LB, ltime = localsearch(n1, m2, s1, s2)