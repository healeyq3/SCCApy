import unittest

import numpy as np

from sccapy.tree.node import Node
from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData
# from scc

n1, n2 = 20, 20
s1, s2 = 10, 10
prob: ProblemData = ProblemData(n1, n2, s1, s2)
A, B, C = gen_data(n1, n2, s1, s2)
setattr(prob, "A", np.matrix(np.copy(A)))
setattr(prob, "B", np.matrix(np.copy(B)))
setattr(prob, "C", np.matrix(np.copy(C)))
almost_full_S0_1 = list(range(s1, n1-s1)) + list(range(n1+s2, n2-s2-1))
almost_full_S1_1 = list(range(s1)) + list(range(n1, n1 + s2 - 1))
full_S1_1 = list(range(s1)) + list(range(n1, n1 + s2))

# node1: Node = 

class TestNode(unittest.TestCase):

    """
    Test variable fixing...although this should probably be in the file where I test the tree functions
    """
    
    def test_attributes(self):
        pass

    def test_properties(self):
        pass


if __name__ == "__main__":
    
    unittest.main()