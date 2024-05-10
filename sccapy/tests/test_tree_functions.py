import unittest
from copy import deepcopy

import numpy as np

from sccapy.tree.tree import (Tree, BranchStrategy)
from sccapy.tree.node import Node

from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData
from sccapy.utilities.bounding_func import (Bounder, LowerBounder)
from sccapy.utilities.objective_func import Objective

from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.variable_fix import varfix
from sccapy.research_algorithms.upper_bound import upper_bound
from sccapy.research_algorithms.objective_eval import fval

n1, n2 = 20, 20
s1, s2 = 10, 10
prob: ProblemData = ProblemData(n1, n2, s1, s2)
A, B, C = gen_data(n1, n2, s1, s2)
setattr(prob, "A", np.matrix(np.copy(A)))
setattr(prob, "B", np.matrix(np.copy(B)))
setattr(prob, "C", np.matrix(np.copy(C)))

phi_ub: Bounder = Bounder(prob, upper_bound)
phi_lb: LowerBounder = LowerBounder(prob, localsearch)
lb, _ = phi_lb()
obj: Objective = Objective(prob, fval)

_, fixed_vars, var_scores, _ = varfix(prob, lb)

tree: Tree = Tree(n1=n1, n2=n2, s1=s1, s2=s2, phi_ub=phi_ub, phi_lb=phi_lb, obj=obj)
tree.branch_strategy = BranchStrategy.SHRINK

class TestTree(unittest.TestCase):

    def test_fix_var(self):
        s1_count, s2_count = tree._fix_vars(fixed_vars)
        self.assertEqual(s1_count, 5)
        self.assertEqual(s2_count, 4)

    def test_root_creation(self):
        tree._create_root_node([], deepcopy(fixed_vars), 5, 4)

        ub, _ = phi_ub([], fixed_vars)
        lb, _ = phi_lb()

        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(tree.gap, ub-lb)
        self.assertEqual(tree.initial_gap, ub - lb)

    def test_create_var_scores_prime(self):
        """
        create the two terminal leafs and make sure list contains no values
        from the terminal leafs...or rather, they are math.inf

        also make sure 
        """
        pass

if __name__ == "__main__":
    
    unittest.main()