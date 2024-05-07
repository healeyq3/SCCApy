import math
from typing import List

class Node:
    
    num_instances: int = 0

    def __init__(self, fixed_in: List[int], fixed_out: List[int], s1_prime: int,
                 s2_prime: int, l1_prime: int, l2_prime: int) -> None:
        Node.num_instances += 1
        self.num_node = Node.num_instances # make sure this does what you think it does. TODO:unittest

        self.fixed_in: List[int] = fixed_in
        self.fixed_out: List[int] = fixed_out
        
        self.s1_prime : int = s1_prime
        self.s2_prime : int = s2_prime
        self.l1_prime : int = l1_prime
        self.l2_prime : int = l2_prime
        self._ub: float = math.inf

    def __eq__(self, other):
        return self._ub == other._ub
    
    def __lt__(self, other):
        return self._ub < other._ub
    
    @property
    def ub(self):
        return self._ub
    
    @ub.setter
    def ub(self, value):
        self._ub = value