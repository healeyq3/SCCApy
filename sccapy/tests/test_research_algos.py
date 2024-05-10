import unittest
from numbers import Number
import math

import numpy as np

from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData
from sccapy.research_algorithms.upper_bound import upper_bound
from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.objective_eval import fval
from sccapy.research_algorithms.variable_fix import varfix

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

n12, n22 = 40, 40
s12, s22 = 15, 15
prob2: ProblemData = ProblemData(n12, n22, s12, s22)
A2, B2, C2 = gen_data(n12, n22, s12, s22)
setattr(prob2, "A", np.matrix(np.copy(A2)))
setattr(prob2, "B", np.matrix(np.copy(B2)))
setattr(prob2, "C", np.matrix(np.copy(C2)))
almost_full_S0_2 = list(range(s12, n12-s12)) + list(range(n12+s22, n22-s22-1))
almost_full_S1_2 = list(range(s12)) + list(range(n12, n12 + s22 - 1))
full_S1_2 = list(range(s12)) + list(range(n12, n12 + s22))

n13, n23 = 100, 100
prob3 = ProblemData = ProblemData(n13, n23, s12, s22)
A3, B3, C3 = gen_data(n13, n23, s12, s22)
setattr(prob3, "A", np.matrix(np.copy(A3)))
setattr(prob3, "B", np.matrix(np.copy(B3)))
setattr(prob3, "C", np.matrix(np.copy(C3)))
almost_full_S0_3 = list(range(s1, n13-s1)) + list(range(n13+s2, n23-s2-1))
almost_full_S1_3 = list(range(s1)) + list(range(n13, n13 + s2 - 1))
full_S1_3 = list(range(s1)) + list(range(n13, n13 + s2))

class TestResearchAlgos(unittest.TestCase):

    def test_upper_bounding_working(self):

        '''
        Cases)
        i. empty lists
        ii. one empty one not
        iii. both nonempty
        iv. one full the other not
        '''

        try:
            bound, time = upper_bound(prob, [], [])
            self.assertIsInstance(bound, Number, "Both empty: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "Both empty: The runtime should be a Number")
            self.assertGreater(time, 0, "Both empty: The runtime should be positive")
        except:
            self.fail("upper_bound method call with empty lists caused"\
                      " an exception to be raised")
            
        try:
            bound, time = upper_bound(prob, [0, 4, 25, 35], [])
            self.assertIsInstance(bound, Number, "S0 nonempty: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "S0 nonempty: The runtime should be a Number")
            self.assertGreater(time, 0, "S0 nonempty: The runtime should be positive")
        except:
            self.fail("upper_bound method call with S0 nonempty and S1 empty caused\
                      an exception to be raised")
            
        try:
            bound, time = upper_bound(prob, [], [1, 10, 20, 34])
            self.assertIsInstance(bound, Number, "S1 nonempty: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "S1 nonempty: The runtime should be a Number")
            self.assertGreater(time, 0, "S1 nonempty: The runtime should be positive")
        except:
            self.fail("upper_bound method call with S0 empty and S1 nonempty caused\
                      an exception to be raised") 
        
        try:
            bound, time = upper_bound(prob, [0, 4, 25, 35], [1, 10, 20, 34])
            self.assertIsInstance(bound, Number, "Both nonempty: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "both nonempty: The runtime should be a Number")
            self.assertGreater(time, 0, "both nonempty: The runtime should be positive")
        except:
            self.fail("upper_bound method call with both nonempty\
                      an exception to be raised")
            
        # Note that "full" means as full as possible before the obj_fn will be called
        # by the tree
        
        try:
            bound, time = upper_bound(prob, almost_full_S0_1, [])
            self.assertIsInstance(bound, Number, "S0 full: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "S0 full: The runtime should be a Number")
            self.assertGreater(time, 0, "S0 full: The runtime should be positive")
        except:
            self.fail("upper_bound method call with S0 full and S1 empty caused\
                      an exception to be raised")
            
        try:
            bound, time = upper_bound(prob, [], almost_full_S1_1)
            self.assertIsInstance(bound, Number, "S1 full: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "S1 full: The runtime should be a Number")
            self.assertGreater(time, 0, "S1 full: The runtime should be positive")
        except:
            self.fail("upper_bound method call with S0 empty and S1 full caused\
                      an exception to be raised")
            
        try:
            bound, time = upper_bound(prob, almost_full_S0_1, almost_full_S1_1)
            self.assertIsInstance(bound, Number, "Both full: The upper bound should be a Number")
            self.assertIsInstance(time, Number, "Both full: The runtime should be a Number")
            self.assertGreater(time, 0, "Both full: The runtime should be positive")
        except:
            self.fail("upper_bound method call with S0 empty and S1 full caused\
                      an exception to be raised")
        

    def test_upper_bound_values(self):
        '''
        No errors should be thrown since we checked that above

        Just ensure the bound shrinks

        '''
        b1, _ = upper_bound(prob3, [], [])
        # b2, _ = upper_bound(prob, [0, 4, 25, 35], [])
        b3, _ = upper_bound(prob3, almost_full_S0_3, almost_full_S1_3)
        self.assertLessEqual(b3, b1, "b2 !<= b1")


    def test_feasible_generator(self):
        '''
        Just make sure it doesn't error for different problem instances
        '''
        try:
            bound, time = localsearch(prob)
            self.assertIsInstance(bound, Number, "The lower bound should be a Number")
            self.assertIsInstance(time, Number, "The runtime should be a Number")
            self.assertGreater(time, 0, "The runtime should be positive")
        except:
            self.fail("localsearch method call with prob failed")

        try:
            bound, time = localsearch(prob2)
            self.assertIsInstance(bound, Number, "The lower bound should be a Number")
            self.assertIsInstance(time, Number, "The runtime should be a Number")
            self.assertGreater(time, 0, "The runtime should be positive")
        except:
            self.fail("localsearch method call with prob2 failed")

        try:
            bound, time = localsearch(prob3)
            self.assertIsInstance(bound, Number, "The lower bound should be a Number")
            self.assertIsInstance(time, Number, "The runtime should be a Number")
            self.assertGreater(time, 0, "The runtime should be positive")
        except:
            self.fail("localsearch method call with prob3 failed")
    
    def test_upper_lower_comparison(self):
        ub1, _ = upper_bound(prob, [], [])
        lb1, _ = localsearch(prob)
        self.assertGreaterEqual(ub1, lb1, "For prob data ub !>= lb")

        ub2, _ = upper_bound(prob2, [], [])
        lb2, _ = localsearch(prob2)
        self.assertGreaterEqual(ub2, lb2, "For prob data2 ub !>= lb")

        ub3, _ = upper_bound(prob3, [], [])
        lb3, _ = localsearch(prob3)
        self.assertGreaterEqual(ub3, lb3, "For prob data3 ub !>= lb")

    
    def test_objective_evaluator(self):
        '''
        Only need to test when leaves are full.

        Test for a few different problem instances
        '''
        self.assertListEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                             full_S1_1)

        try:
            val, time = fval(prob, full_S1_1)
            self.assertIsInstance(val, Number, "prob1: val should be a Number")
            self.assertIsInstance(time, Number, "prob1: The runtime should be a Number")
            self.assertGreater(time, 0, "prob1: The runtime should be positive")
        except:
            self.fail("fval method call with prob caused an exception to be raised")

        try:
            val, time = fval(prob2, full_S1_2)
            self.assertIsInstance(val, Number, "prob2: val should be a Number")
            self.assertIsInstance(time, Number, "prob2: The runtime should be a Number")
            self.assertGreater(time, 0, "prob2: The runtime should be positive")
        except:
            self.fail("fval method call with prob2 caused an exception to be raised")

        try:
            val, time = fval(prob3, S1=full_S1_3)
            self.assertIsInstance(val, Number, "prob3: val should be a Number")
            self.assertIsInstance(time, Number, "prob3: The runtime should be a Number")
            self.assertGreater(time, 0, "prob3: The runtime should be positive")
        except:
            self.fail("fval method call with prob3 caused an exception to be raised")  

    def test_variable_fixing(self):
        '''
        Ensure that none of the scores are math.inf

        check for a few different problem instances.
        '''
        lb1, _ = localsearch(prob)
        lb2, _ = localsearch(prob2)
        lb3, _ = localsearch(prob3)

        try:
            _, fixed, scores, runtime = varfix(prob, lb1)
            for fix in fixed:
                self.assertIsInstance(fix, int, "prob1: fixed variable is not an integer")
            self.assertIsInstance(runtime, Number, "prob1: The runtime should be a Number")
            self.assertGreater(runtime, 0, "prob1: The runtime should be positive")
            self.assertEqual(len(scores), n1+n2, "prob1: not enough variable scores")
            for score in scores:
                self.assertLess(score, math.inf, "prob1: score is infinity")
        except:
            self.fail("varfix method call with prob1 caused an exception to be raised")

        try:
            _, fixed, scores, runtime = varfix(prob2, lb2)
            for fix in fixed:
                self.assertIsInstance(fix, int, "prob2: fixed variable is not an integer")
            self.assertIsInstance(runtime, Number, "prob2: The runtime should be a Number")
            self.assertGreater(runtime, 0, "prob2: The runtime should be positive")
            self.assertEqual(len(scores), n12+n22, "prob2: not enough variable scores")
            for score in scores:
                self.assertLess(score, math.inf, "prob2: score is infinity")
        except:
            self.fail("varfix method call with prob2 caused an exception to be raised")

        try:
            _, fixed, scores, runtime = varfix(prob3, lb3)
            for fix in fixed:
                self.assertIsInstance(fix, int, "prob3: fixed variable is not an integer")
            self.assertIsInstance(runtime, Number, "prob3: The runtime should be a Number")
            self.assertGreater(runtime, 0, "prob3: The runtime should be positive")
            self.assertEqual(len(scores), n13+n23, "prob3: not enough variable scores")
            for score in scores:
                self.assertLess(score, math.inf, "prob3: score is infinity")
        except:
            self.fail("varfix method call with prob3 caused an exception to be raised")

if __name__ == "__main__":
    
    unittest.main()
