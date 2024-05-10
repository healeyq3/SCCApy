import unittest
from copy import deepcopy

import numpy as np

from sccapy.tree.node import Node
from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData

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

Node.n1 = n1
Node.n2 = n2
Node.s1 = s1
Node.s2 = s2

almost_full_S0_ = list(range(s1, n1-s1)) + list(range(n1+s2, n2-s2-1))
almost_full_S1 = list(range(s1)) + list(range(n1, n1 + s2 - 1))
full_S1 = list(range(s1)) + list(range(n1, n1 + s2))
# full_S0 = 

# both 
internal_node_S1 = [0, 3] + [22, 23]
internal_node_S0 = list(np.random.randint(5, n1, size=5)) + list(np.random.randint(n1+5, n1 + n2, size=5))
internal_node: Node = Node(fixed_in=internal_node_S1, fixed_out=internal_node_S0,
                           s1_prime=2, s2_prime=2, l1_prime=5, l2_prime=5)
internal_node.ub = 20


x_terminal_S1 = list(range(10)) + [22, 23]
x_terminal_S0 = list(np.random.randint(5, n1, size=5)) + list(np.random.randint(n1+5, n1 + n2, size=5))
x_terminal_node: Node = Node(fixed_in=x_terminal_S1, fixed_out=x_terminal_S0, s1_prime=10,
                             s2_prime=2, l1_prime=5, l2_prime=5)
x_terminal_node.ub = 15

y_terminal_node_S1  =[0, 3] + [20, 21]
y_terminal_node_S0 = list(np.random.randint(5, n1, size=5)) +\
                    list(range(25, 35)) #list(np.random.randint(n1+5, n1 + n2, size=10))
y_terminal_node: Node = Node(fixed_in=y_terminal_node_S1, fixed_out=y_terminal_node_S0, s1_prime=2,
                             s2_prime=2, l1_prime=5, l2_prime=10)
y_terminal_node.ub = 14

terminal_node: Node = Node(fixed_in=x_terminal_S1, fixed_out=y_terminal_node_S0, s1_prime=10,
                           s2_prime=2, l1_prime=5, l2_prime=10)
terminal_node.ub = 10

class TestNode(unittest.TestCase):

    def test_global_attributes(self):
        self.assertEqual(n1, Node.n1, "n1 not equal")
        self.assertEqual(n2, Node.n2, "n2 not equal")
        self.assertEqual(s1, Node.s1, "s1 not equal")
        self.assertEqual(s2, Node.s2, "s2 not equal")

        self.assertEqual(Node.num_instances, 4, "total node count is incorrect")
    
    def test_internal_node(self):
        self.assertEqual(internal_node.ub, 20, "upper bound not properly set")
        self.assertEqual(internal_node.num_node, 1, "node counter isn't right")

        self.assertEqual(internal_node.s1_prime, 2, "internal node should have 2 selected x variables")
        self.assertEqual(internal_node.s2_prime, 2, "internal node should have 2 selected y variables")
        self.assertEqual(internal_node.l1_prime, 5, "internal node should have 5 discarded x variables")
        self.assertEqual(internal_node.l2_prime, 5, "internal node should have 5 discarded y variables")
        
        self.assertTrue(internal_node.is_x_internal_node, "internal node didn't register as x internal")
        self.assertTrue(internal_node.is_y_internal_node, "internal node didn't register as x internal")

        self.assertFalse(internal_node.is_x_terminal_leaf, "internal node shouldn't register as x terminal")
        self.assertFalse(internal_node.is_y_terminal_leaf, "internal node shouldn't register as y terminal")
        self.assertFalse(internal_node.is_terminal_leaf, "internal node shouldn't register as a leaf")

    def test_x_terminal(self):
        self.assertEqual(x_terminal_node.num_node, 2, "node counter isn't right")

        self.assertTrue(x_terminal_node.is_x_terminal_leaf, "x terminal node didn't register as x terminal")
        self.assertTrue(x_terminal_node.is_y_internal_node, "x terminal node did not identify as y internal")

        self.assertFalse(x_terminal_node.is_y_terminal_leaf, "x terminal node registered as y terminal")
        self.assertFalse(x_terminal_node.is_terminal_leaf, "x terminal leaf registered as a terminal leaf")
        self.assertFalse(x_terminal_node.is_x_internal_node, "x terminal node registered as x internal")

    def test_y_terminal(self):
        self.assertEqual(y_terminal_node.num_node, 3, "node counter isn't right")

        self.assertTrue(y_terminal_node.is_y_terminal_leaf, "y terminal node didn't register as y terminal")
        self.assertTrue(y_terminal_node.is_x_internal_node, "y terminal node did not identify as x internal")

        self.assertFalse(y_terminal_node.is_x_terminal_leaf, "y terminal node registered as x terminal")
        self.assertFalse(y_terminal_node.is_terminal_leaf, "y terminal leaf registered as a terminal leaf")
        self.assertFalse(y_terminal_node.is_y_internal_node, "y terminal node registered as y internal")
    
    def test_terminal_leaf(self):
        self.assertEqual(terminal_node.num_node, 4, "node counter isn't right")

        self.assertTrue(terminal_node.is_terminal_leaf, "terminal node didn't identify as terminal leaf")
        self.assertTrue(terminal_node.is_x_terminal_leaf, "terminal leaf has x terminal")
        self.assertTrue(terminal_node.is_y_terminal_leaf, "terminal leaf has y terminal")

        self.assertFalse(terminal_node.is_x_internal_node, "terminal node incorrectly identified as having x free")
        self.assertFalse(terminal_node.is_y_internal_node, "terminal node incorrectly identified as having y free")
        
        selected_vars_correct = list(range(10)) + [20, 21, 22, 23, 24] + [35, 36, 37, 38, 39]
        self.assertCountEqual(selected_vars_correct, terminal_node.feasible_solution)

    # test more solution generations
    def test_remaining_solution_generations(self):
        S1 = [i for i in list(range(n1)) if i % 2 == 0] + [i for i in list(range(n1, n1+n2)) if i % 2 == 1]
        # [i for i in list(range(20)) if i % 2 == 0] + [i for i in list(range(20, 40)) if i % 2 == 1]
        S0 = []
        
        new_node: Node = Node(fixed_in=deepcopy(S1), fixed_out=deepcopy(S0), s1_prime=10, s2_prime=10,
                              l1_prime=0, l2_prime=0)
        
        self.assertTrue(new_node.is_terminal_leaf)
        self.assertCountEqual(new_node.feasible_solution, S1)

        S1 = []
        S0 = [i for i in list(range(n1)) if i % 2 == 1] + [i for i in list(range(n1, n1+n2)) if i % 2 == 0]
        correct_solution = [i for i in list(range(n1)) if i % 2 == 0] + [i for i in list(range(n1, n1+n2)) if i % 2 == 1]

        new_node = Node(fixed_in=deepcopy(S1), fixed_out=deepcopy(S0), s1_prime=0, s2_prime=0,
                        l1_prime=10, l2_prime=10)
        
        self.assertTrue(new_node.is_terminal_leaf)
        self.assertCountEqual(new_node.feasible_solution, correct_solution)

        S1 = [10, 11] + list(range(25, 25+s2))
        S0 = [5, 6, 7, 8, 9] + [15, 16, 17, 18, 19] + [22, 34]
        correct_solution = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14] + list(range(25, 25+s2))

        new_node = Node(fixed_in=deepcopy(S1), fixed_out=deepcopy(S0), s1_prime=2, s2_prime=10,
                        l1_prime=10, l2_prime=2)
        
        self.assertTrue(new_node.is_terminal_leaf)
        self.assertCountEqual(new_node.feasible_solution, correct_solution)
    
    def test_node_comparison(self):
        nodes: list[Node] = [x_terminal_node, y_terminal_node, internal_node, terminal_node]
        
        self.assertEqual(np.argmax(nodes), 2)
        nodes.pop(np.argmax(nodes))

        self.assertEqual(np.argmax(nodes), 0)
        nodes.pop(np.argmax(nodes))

        self.assertEqual(np.argmax(nodes), 0)
        nodes.pop(np.argmax(nodes))

        self.assertEqual(np.argmax(nodes), 0)
        nodes.pop(np.argmax(nodes))


if __name__ == "__main__":
    
    unittest.main()