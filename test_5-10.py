import numpy as np

import sccapy
from sccapy.main import Problem

from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData

from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.objective_eval import fval
from sccapy.research_algorithms.variable_fix import varfix
from sccapy.research_algorithms.upper_bound import upper_bound

n1, n2 = 60, 60
s1, s2 = 5, 5
prob: ProblemData = ProblemData(n1, n2, s1, s2)
A, B, C = gen_data(n1, n2, s1, s2)
setattr(prob, "A", np.matrix(np.copy(A)))
setattr(prob, "B", np.matrix(np.copy(B)))
setattr(prob, "C", np.matrix(np.copy(C)))

lb, _ = localsearch(prob)

_, fixed_vars, var_scores, var_fixing_time = varfix(prob, lb)

ub, _ = upper_bound(prob, [], fixed_vars)

# print("OUTSIDE CALL GAP: ", ub - lb) # DEBUG

prob1: Problem = Problem(n1, n2, s1, s2, A=A, B=B, C=C)

prob1.solve(lower_bound_func=localsearch, upper_bound_func=upper_bound,
            var_fix=varfix, objective_func=fval)