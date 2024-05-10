from numbers import Number
import unittest

import numpy as np

from sccapy.utilities.problem_data import ProblemData
from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.objective_func import Objective
from sccapy.utilities.bounding_func import (Bounder, LowerBounder)
from sccapy.research_algorithms.objective_eval import fval
from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.upper_bound import upper_bound


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

class TestBounder(unittest.TestCase):

    '''
    Just need to ensure the Bounder, LowerBounder, and Objective
    are properly wrapping around the research algorithms.
    
    '''

    def test_lower_bounder(self):
        try:
            phi_lb = LowerBounder(prob, localsearch)
            lb, time = phi_lb()
            lb_true, _ = localsearch(prob)
            self.assertIsInstance(lb, Number, "Lower Bounder prob: The lower bound should be a Number.")
            self.assertIsInstance(time, Number, "Lower Bounder prob: The runtime should be a Number.")
            self.assertGreater(time, 0, "Lower Bounder prob: The runtime should be positive.")
            self.assertEqual(lb, lb_true, "Lower Bound prob: wrapper returned incorrect value")
        except:
            self.fail("LowerBounder failed with prob data")

        try:
            phi_lb = LowerBounder(prob2, localsearch)
            lb, time = phi_lb()
            lb_true, _ = localsearch(prob2)
            self.assertIsInstance(lb, Number, "Lower Bounder prob2: The lower bound should be a Number.")
            self.assertIsInstance(time, Number, "Lower Bounder prob2: The runtime should be a Number.")
            self.assertGreater(time, 0, "Lower Bounder prob2: The runtime should be positive.")
            self.assertEqual(lb, lb_true, "Lower Bound prob: wrapper returned incorrect value")
        except:
            self.fail("LowerBounder failed with prob2 data")

        try:
            phi_lb = LowerBounder(prob3, localsearch)
            lb, time = phi_lb()
            lb_true, _ = localsearch(prob3)
            self.assertIsInstance(lb, Number, "Lower Bounder prob: The lower bound should be a Number.")
            self.assertIsInstance(time, Number, "Lower Bounder prob: The runtime should be a Number.")
            self.assertGreater(time, 0, "Lower Bounder prob: The runtime should be positive.")
            self.assertEqual(lb, lb_true, "Lower Bound prob: wrapper returned incorrect value")
        except:
            self.fail("Bounder failed with prob data")

    def test_upper_bounder_working(self):
        try:
            phi_ub = Bounder(prob, upper_bound)
            ub, time = phi_ub([], [])
            self.assertIsInstance(ub, Number, "Upper Bounder prob: The upper bound should be a Number.")
            self.assertIsInstance(time, Number, "Upper Bounder prob: The runtime should be a Number.")
            self.assertGreater(time, 0, "Upper Bounder prob: The runtime should be positive.")
        except:
            self.fail("Bounder failed with prob data")

        try:
            phi_ub = Bounder(prob2, upper_bound)
            ub, time = phi_ub(almost_full_S0_2, almost_full_S1_2)
            self.assertIsInstance(ub, Number, "Upper Bounder prob2: The upper bound should be a Number.")
            self.assertIsInstance(time, Number, "Upper Bounder prob2: The runtime should be a Number.")
            self.assertGreater(time, 0, "Upper Bounder prob2: The runtime should be positive.")
            # self.assertEqual(ub, ub_true, "Upper Bound prob2: wrapper returned incorrect value")
        except:
            self.fail("Bounder failed with prob2 data")
        
        try:
            phi_ub = Bounder(prob3, upper_bound)
            ub, time = phi_ub(almost_full_S0_3, [])
            self.assertIsInstance(ub, Number, "Upper Bounder prob3: The upper bound should be a Number.")
            self.assertIsInstance(time, Number, "Upper Bounder prob3: The runtime should be a Number.")
            self.assertGreater(time, 0, "Upper Bounder prob3: The runtime should be positive.")
        except:
            self.fail("Bounder failed with prob3 data")

    def test_upper_bounder_proper_vals(self):
        
        phi_ub = Bounder(prob, upper_bound)
        ub, time = phi_ub([], [])
        ub_true, _ = upper_bound(prob, [], [])
        self.assertEqual(ub, ub_true, "Upper Bound prob: wrapper returned incorrect value")

        phi_ub = Bounder(prob2, upper_bound)
        ub, time = phi_ub(almost_full_S0_2, almost_full_S1_2)
        ub_true, _ = upper_bound(prob2, almost_full_S0_2, almost_full_S1_2)
        self.assertEqual(ub, ub_true, "Upper Bound prob2: wrapper returned incorrect value")

        phi_ub = Bounder(prob3, upper_bound)
        ub, time = phi_ub(almost_full_S0_3, [])
        ub_true, _ = upper_bound(prob3, almost_full_S0_3, [])
        self.assertEqual(ub, ub_true, "Upper Bound prob3: wrapper returned incorrect value")
    
    ### Objective evaluator uses same configuration. Test later. Need to move on for now. ###


if __name__ == "__main__":
    
    unittest.main()