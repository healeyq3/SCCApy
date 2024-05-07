from typing import List, Tuple
import math
import time

import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData
from sccapy.research_algorithms.greedy import greedy

def localsearch(data: ProblemData):
    start_time = time.time()

    n1, n2 = data.n1, data.n2
    s1, s2 = data.s1, data.s2
    A, B, C = data.A, data.B, data.C

    S1, S2, bestf, _ = greedy(data)
    
    sn = list(range(n1))
    sm = list(range(n2))
    
    optimal = False

    unsel = []
    unsel = list(set(sn) - set(S1)) 
    
    unsel2 = []
    unsel2 = list(set(sm) - set(S2))
    
    while(optimal == False):
        optimal = True
        
        # first update row selection
        temp2 = C[ix_(S2, S2)]
        temp2 = matrix(LA.sqrtm(temp2.I))
        
        for i in S1:
            for j in unsel:
                S1.remove(i)
                S1.append(j)
                
                temp1 = B[ix_(S1, S1)]
                temp1 = matrix(LA.sqrtm(temp1.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
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
                

        temp1 = B[ix_(S1, S1)]
        temp1 = matrix(LA.sqrtm(temp1.I))
        for i in S2:
            for j in unsel2:
                S2.remove(i)
                S2.append(j)

                temp2 = C[ix_(S2, S2)]
                temp2 = matrix(LA.sqrtm(temp2.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 


                if LB > bestf:
                    print(i)
                    optimal = False                             
                    bestf = LB
                    unsel2.append(i)
                    unsel2.remove(j)
                    break
                
                S2.append(i)
                S2.remove(j)
                
    end = time.time()
    total_time = end - start_time
    
    print('the lower bound at the root node is ',  bestf) 
    
    return bestf, total_time