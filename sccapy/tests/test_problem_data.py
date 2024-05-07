import unittest
import numpy as np

from sccapy.utilities.problem_data import ProblemData

class TestProblemData(unittest.TestCase):
    # inheriting gives us access to different testing capabilities within the class

    # this naming convention is required, otherwise the test won't be run
    def test_problem_data_creation_invalid(self):

        with self.assertRaises(AssertionError):
            ProblemData(40, 40, 45, 25)
            ProblemData(40, 40, 40, 25)
            ProblemData(40, 40, 25, 45)
            ProblemData(40, 40, 25, 40)

    def test_problem_data_creation_basic(self):
        prob1: ProblemData = ProblemData(40, 40, 20, 20)

        self.assertEqual(prob1.n1, 40)
        self.assertEqual(prob1.n2, 40)
        self.assertEqual(prob1.s1, 20)
        self.assertEqual(prob1.s2, 20)

    def test_problem_data_creation_advanced(self):
        prob1: ProblemData = ProblemData(40, 40, 20, 20)

        with self.assertRaises(AttributeError):
            prob1.A

        with self.assertRaises(AttributeError):
            prob1.B

        with self.assertRaises(AttributeError):
            prob1.C

        A = np.random.rand(40, 40)
        B = np.random.rand(40, 40)
        C = np.random.rand(40, 40)

        extra_data = {"A" : np.copy(A),
                      "B" : np.copy(B),
                      "C" : np.copy(C)}
        
        for field, value in extra_data.items():
            setattr(prob1, field, value)

        self.assertIsNotNone(prob1.A)
        self.assertIsNotNone(prob1.B)
        self.assertIsNotNone(prob1.C)

        self.assertTrue(np.array_equal(prob1.A, A)) 
        self.assertTrue(np.array_equal(prob1.B, B))
        self.assertTrue(np.array_equal(prob1.C, C))


if __name__ == "__main__":
    unittest.main()