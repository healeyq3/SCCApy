"""
@author: yongchunli

This algorithm was created by Dr. Xie and Dr. Li. 

I (Quill) just made some syntactical changes and changes to ensure 
the algorithms worked with my framework.
"""

from typing import List, Tuple
import math
import time

import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def greedy(data: ProblemData) -> Tuple[List[int], List[int], float, float]:
    
    start = time.time()
    
    n1, n2 = data.n1, data.n2
    s1, s2 = data.s1, data.s2
    A, B, C = data.A, data.B, data.C
    
    s = min(s1, s2)
    sn = list(range(n1))
    sm = list(range(n2))
    bestf = 0.0
    
    # initialize two subsets
    S1 = []
    S2 = []  
    temp = zeros([n1, n2])
    
    for i in range(n1):
        for j in range(n2):
            temp[i,j] = abs(1/math.sqrt(B[i,i]) * A[i,j] * 1/math.sqrt(C[j,j]))
    
    sindex = unravel_index(argmax(abs(temp), axis=None), temp.shape)
    
    S1.append(sindex[0])
    S2.append(sindex[1])    
    
    for i in range(s-1):
        unsel = []
        unsel = list(set(sn) - set(S1)) 
        bestLB = 0.0
        bestl = 0
   
        temp2 = C[ix_(S2, S2)]
        temp2 = matrix(LA.sqrtm(temp2.I))
        
        for l in unsel:
            S1.append(l)  
            temp1 = B[ix_(S1, S1)]
            temp1 = matrix(LA.sqrtm(temp1.I))
            
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
            
            LB = max(sigma)
            
            if LB > bestLB:
                bestLB = LB
                bestl = l
            S1.remove(l)
            
        S1.append(bestl)
        temp1 = B[ix_(S1, S1)]
        temp1 = matrix(LA.sqrtm(temp1.I))
        
        # then update set S2    
        unsel = []
        unsel = list(set(sm) - set(S2)) 
        bestLB = 0.0
        bestl = 0
        for l in unsel:
            S2.append(l)
            temp2 = C[ix_(S2, S2)]
            temp2 = matrix(LA.sqrtm(temp2.I))
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 

            LB = 0.0
            LB =max(sigma)

            if LB > bestLB:
                bestLB = LB
                bestl = l
            S2.remove(l)
            
        S2.append(bestl)
        
    if s1 < s2:
        temp1 = B[ix_(S1, S1)]

        temp1 = matrix(LA.sqrtm(temp1.I))
        
        for i in range(s, s2):
            # only update set S2    
            unsel = []
            unsel = list(set(sm) - set(S2)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S2.append(l)
                temp2 = C[ix_(S2, S2)]
                temp2 = matrix(LA.sqrtm(temp2.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 

                LB = 0.0
                LB =max(sigma)

                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S2.remove(l)
                
            S2.append(bestl)
    else:
        
        temp2 = C[ix_(S2, S2)]
        temp2 = matrix(LA.sqrtm(temp2.I))
        
        for i in range(s, s1):
            # only update set S1
            unsel = []
            unsel = list(set(sn) - set(S1)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S1.append(l)  
                temp1 = B[ix_(S1, S1)]
                temp1 = matrix(LA.sqrtm(temp1.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                    
                S1.remove(l)
                
            S1.append(bestl)
            
    end = time.time()
    total_time = (end-start)
    bestf = bestLB
    
    return  S1, S2, bestf, total_time  