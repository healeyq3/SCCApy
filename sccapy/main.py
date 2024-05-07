from typing import Callable, List, Tuple
# from sccapy.utilities.problem_data import ProblemData
from utilities.problem_data import ProblemData

class Problem:

    def __init__(self, n1: int, n2: int, s1: int, s2: int,
                
                 **kwargs) -> None:
        
        # Problem data has checks for the integers 
        self.data = ProblemData(n1, n2, s1, s2)

        for key, value in kwargs.items():
            # do a validity check here
            setattr(self.data, key, value)

        # keep track of whether solve has been attempted or not.

    def solve(eps: float, timeout: float,
                lower_bound_func: Callable[[ProblemData, List[int], List[int]],
                                            Tuple[float, float]] = None,
                upper_bound_func: Callable[[ProblemData, List[int], List[int]],
                                            Tuple[float, float]] = None,
                objective_func: Callable[[ProblemData, List[int], List[int]],
                                            Tuple[float, float]] = None,
                var_fix: Callable[[ProblemData, float],
                                            Tuple[List[int], List[int], List[float]]] = None,):
        pass

    
    def generate_statistics():
        pass

if __name__ == '__main__':
    print("SUCCESS")