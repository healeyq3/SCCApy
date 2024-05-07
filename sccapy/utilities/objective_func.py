from typing import List, Tuple

from utilities.problem_data import ProblemData

class Objective:

    def __init__(self, data: ProblemData, proposed_func) -> None:
        self.data = data
        self._test_func(proposed_func)
        self.f0 = proposed_func

    def __call__(self, S0: List[int]=[], S1: List[int]=[]) -> Tuple[float, float]:
        return self.f0(self.data, S0, S1)
    
    def _test_func(proposed_func):
        '''
        Try unit tests here
        '''
        pass