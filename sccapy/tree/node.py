import math
from typing import List


class Node:
    num_instances: int = 0

    n1: int
    n2: int
    s1: int
    s2: int

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
        self._ub: float = math.inf # also is obj_val if this is a terminal leaf node

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

    @property
    def s1_prime_full(self) -> bool:
        return self.s1_prime == Node.s1
    
    @property
    def s2_prime_full(self) -> bool:
        return self.s2_prime == Node.s2
    
    @property
    def l1_prime_full(self) -> bool:
        return self.l1_prime == Node.n1 - Node.s1 - self.s1_prime
    
    @property
    def l2_prime_full(self) -> bool:
        return self.l2_prime == Node.n2 - Node.s2 - self.s2_prime
    
    @property
    def is_x_internal_node(self) -> bool:
        return not (self.s1_prime_full or self.l1_prime_full)
    
    @property
    def is_x_terminal_leaf(self) -> bool:
        return self.s1_prime_full and self.l1_prime_full
    
    @property
    def is_y_internal_node(self) -> bool:
        return not (self.s2_prime_full or self.l2_prime_full)
    
    @property
    def is_y_terminal_leaf(self) -> bool:
        return self.s2_prime_full and self.l2_prime_full
    
    @property
    def is_terminal_leaf(self) -> bool:
        return self.is_x_terminal_leaf and self.is_y_terminal_leaf