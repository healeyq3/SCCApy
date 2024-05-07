import unittest

from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData
from sccapy.research_algorithms.upper_bound import upper_bound
import sccapy.research_algorithms.local_search

n1, n2 = 40, 40
s1, s2 = 10, 10
prob: ProblemData = ProblemData(n1, n2, s1, s2)
A, B, C = gen_data(n1, n2, s1, s2)

class TestResearchAlgos(unittest.TestCase):
    # inheriting gives us access to different testing capabilities within the class

    # this naming convention is required, otherwise the test won't be run
    def test_upper_bounding_working(self):

        upper_bound(ProblemData)

        '''
        Cases)
        i. empty lists
        ii. one empty one not
        iii. 
        '''