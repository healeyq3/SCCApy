"""
@author: yongchunli

This algorithm was created by Dr. Xie and Dr. Li. 

I (Quill) just made some syntactical changes and changes to ensure 
the algorithms worked with my framework.
"""

from typing import Tuple
import random
from random import sample
import math

import numpy as np

def gen_data(n1, n2, s1, s2) -> Tuple[np.matrix, np.matrix, np.matrix]:
    
    np.random.seed(1)
    random.seed(1)
    
    B = np.matrix(np.random.normal(0, 1, (n1, n1)))
    B = B*B.T + np.matrix(np.eye(n1, dtype=float))

    C = np.matrix(np.random.normal(0, 1, (n2, n2)))
    C = C*C.T + np.matrix( np.eye(n2, dtype=float)) 
    
    NS1 = sample(list(range(n1)), n1-s1)
    NS2 = sample(list(range(n2)), n2-s2)
    
    u = np.matrix(np.random.uniform(0, 1, (n1)))
    for i in NS1:
        u[0,i] = 0   
    u = u/math.sqrt((u*B*u.T)[0,0])
    
    v = np.matrix(np.random.uniform(0, 1, (n2)))
    for i in NS2:
        v[0,i] = 0   
    v = v/math.sqrt((v*C*v.T)[0,0])
    
    l = np.random.uniform(0,1)

    A = l*B*u.T*v*C
    
    cov = np.block([[B, A], [A.T, C]])
    mean = [0.0]*(n1+n2)
    
    N = 5000
    np.random.seed(1)
    temp = np.random.multivariate_normal(mean, cov, N)
    X = np.matrix(temp[:,0:n1])
    Y = np.matrix(temp[:, n1:(n1+n2)])
    
    B = sum([X[i,:].T*X[i,:] for i in range(N)])/N
    C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N
    A = sum([X[i,:].T*Y[i,:] for i in range(N)])/N

    return (A, B, C)