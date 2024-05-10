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

def varfix(data: ProblemData, LB: float) -> Tuple[List[int], List[int], List[float], float]:
    start_time = time.time()
    n1, n2 = data.n1, data.n2
    A, B, C = data.A, data.B, data.C

    scores=[]
    OptS1 = []
    sn = list(range(n1))
    sm = list(range(n2))
    temp2 = matrix(LA.sqrtm(C.I))
    for i in range(n1):
        sn.remove(i)
        temp1 = B[ix_(sn, sn)]
        temp1 = matrix(LA.sqrtm(temp1.I))
        _, sigma, _ = svd(temp1 * A[ix_(sn, sm)] * temp2)
        sn.append(i)
        scores.append(max(sigma))
        if max(sigma) < LB:
            # print('S1:', i, max(sigma), LB)
            OptS1.append(i)
            
    OptS2 = []
    sn = list(range(n1))
    sm = list(range(n2))
    temp1 = matrix(LA.sqrtm(B.I))
    for i in range(n2):
        sm.remove(i)
        temp2 = C[ix_(sm, sm)]
        temp2 = matrix(LA.sqrtm(temp2.I))
        _, sigma, _ = svd(temp1 * A[ix_(sn, sm)] * temp2)
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
    
    runtime = time.time() - start_time
    return [], S1, scores, runtime