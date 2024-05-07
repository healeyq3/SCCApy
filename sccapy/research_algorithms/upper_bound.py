"""
@author: yongchunli

This algorithm was created by Dr. Xie and Dr. Li. 

I (Quill) just made some syntactical changes and changes to ensure 
the algorithms worked with my framework.
"""

from typing import List, Tuple
import time

import scipy.linalg as LA
from numpy import (matrix, ix_)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def upper_bound(data: ProblemData, S0: List[int], S1: List[int]) -> Tuple[float, float]:
    start_time = time.time()
    n1, n2 = data.n1, data.n2
    A, B, C = data.A, data.B, data.C

    zeroS1, zeroS2 = [], []
    for i in S0:
        if i <= n1:
            zeroS1.append(i)
        else:
            zeroS2.append(i-n1)
            
    sn = list(range(n1))
    sm = list(range(n2))
        
    sel1 = []
    sel1 = list(set(sn) - set(zeroS1)) 
    
    sel2 = []
    sel2 = list(set(sm) - set(zeroS2))
    
    temp1 = B[ix_(sel1, sel1)]
    temp1 = matrix(LA.sqrtm(temp1.I))
    temp2 = C[ix_(sel2, sel2)]
    temp2 = matrix(LA.sqrtm(temp2.I))
    _, sigma, _ = svd(temp1 * A[ix_(sel1, sel2)] * temp2)
    end_time = time.time()
    bound_time = end_time - start_time

    return max(sigma), bound_time