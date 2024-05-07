from typing import Tuple, List, Optional
from numbers import Number
import time
import math
from copy import deepcopy

from sccapy.tree.node import Node
from sccapy.bounding.bounding_func import Bounder
from sccapy.utilities.objective_func import Objective

class Tree:

    def __init__(self, n1: int, n2: int, s1: int, s2: int,
                 phi_lb: Bounder, phi_ub: Bounder, obj_fn: Objective) -> None:
        
        self.n1: int = n1
        self.n2: int = n2
        self.s1: int = s1
        self.s2: int = s2
        
        self.phi_lb: Bounder = phi_lb
        self.phi_ub: Bounder = phi_ub
        self.f0 : Objective = obj_fn
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.nodes: List[Node] = []
        self.feasible_leafs: List[Node] = [] # only will be >= initial_LB
        
        ### Research Specific Objects ###
        self.known_fixed_in: Optional[List[int]] = None
        self.variable_scores: Optional[List[float]] = None
        
        ### Framework Metrics ###
        self._status: Optional[str] = None
        self._value = None
        self.num_iter: int = 0
        self.solve_time: float = 0 # total enumeration time
        self.bound_time: float = 0 # total runtime of ub, lb, and obj functions 
        self.initial_gap: float = math.inf
        self.initial_LB: float = -math.inf
        self.initial_UB: float = math.inf

    @property
    def gap(self):
        return self.UB - self.LB 
    
    def solve(self, eps: Number=1e-3, timeout: Number=10, fixed_vars: List[int]=None,
              var_scores: List[float]=None) -> bool:
        """Enumerate a branch and bound tree to solve the SCCA problem to global optimality
        using the bounding and objective functions passed into the tree upon its construction.

        Populates the :code:'status' and :code:'value' attributes on the
        tree object as a side-effect.

        Arguments
        ----------
        eps: float, optional
            The desired optimality tolerance.
            The default tolerance is 1e-3.
        timeout: float, optional
            The number of minutes solve will run before terminating.
            The default timeout is after 60 minutes.
        fixed_vars: List[int], optional
            Variable elements known to be fixed in (i.e. x[i] = 1 forall i in fixed_vars).
        var_scores: List[float], optional
            Scores used to determine variable branching priority.

        Returns
        -------
        bool: Whether or not the problem was solved to global optimality.
        
        Raises
        ------
        AssertionError
            Raised if epsilon or timeout are not Numbers
        ValueError
            Raised if a fixed variable index is negative or >= n1 + n2
        """
        start_time = time.time()
        loop_time = time.time()

        assert isinstance(eps, Number), "eps must be a Number"
        assert isinstance(timeout, Number), "timeout must be a Number"
        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

        # check if there are variable scores -> use argmin later
        if var_scores != None:
            self.variable_scores = deepcopy(var_scores)

        # check if there are fixed variables
        # create root node, gap, LB accordingly
        S0, S1 = [], []
        if fixed_vars != None:
            S1 = fixed_vars
            self._fix_vars(fixed_vars)
        
        self._create_root_node(S0, S1)

        while (self.gap > eps and timeout > (loop_time / 60)):
            node: Node = self.choose_subproblem()

            if (self.num_iter == 0):
                self.LB = self.phi_lb()
            
            # split problem handles updating LB (if possible)
            # and handles adding the new subproblems to nodes (L_k)
            self.split_problem(node)

            self.UB = max(self.nodes)
            
            self.num_iter += 1

            loop_time = time.time() - start_time

            if (self.gap > eps and len(self.nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
        
        if (timeout > loop_time / 60):
            self._status = "solve timed out."
            return False
        
        self._status = "global optimal found."
        return True


    def _fix_vars(self, proposed_fixed_vars):
            s1_count, s2_count = 0, 0
            for i in proposed_fixed_vars:
                if 0 <= i <= self.n1 - 1:
                    s1_count += 1
                elif self.n1 <= i <= self.n1 + self.n2 - 1:
                    s2_count += 1
                else:
                    raise ValueError("fixed variable indices should be nonnegative and less than n1 + n2.")
            self.n1, self.s1 -= s1_count
            self.n2, self.s2 -= s2_count

    def _create_root_node(self, S0, S1):
        root_node: Node = Node(fixed_in=S1, fixed_out=S0, s1_prime=self.s1, s2_prime=self.s2,
                               l1_prime=self.n1-self.s1, l2_prime=self.n1-self.s2)
        
        root_node.ub, root_ub_time = self.phi_ub(root_node.fixed_out, root_node.fixed_in)
        self.bound_time += root_ub_time
        root_lb, root_lb_time = self.phi_lb(root_node.fixed_out, root_node.fixed_in)
        self.bound_time += root_lb_time
        
        self.UB = root_node.ub
        self.LB = root_lb
        self.initial_gap = self.UB - self.LB
        self.nodes.append(root_node)
    
    def choose_subproblem(self) -> Node:
        '''
        either DFS or shrink UB 

        make sure to pop the node from the list
        for DFS just pop most recently added
        for UB use argmax on node list
        '''
        pass

    
    def split_problem(self, node: Node):
        """
        when adding to L_k see if pruning should take place
        
        """
        pass