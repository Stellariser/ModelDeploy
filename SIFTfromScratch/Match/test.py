import unittest
import numpy as np
from utils.evaluation import best_map


class TestBestMap(unittest.TestCase):
    def test_best_map1(self):
        L1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        L2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        newL2 = best_map(L1, L2)
        ans = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertTrue(np.array_equal(newL2, ans))

    def test_best_map2(self):
        L1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        L2 = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2])
        newL2 = best_map(L1, L2)
        ans = np.array([1, 2, 3, 2, 3, 1, 3, 1, 2])
        self.assertTrue(np.array_equal(newL2, ans))

    def test_best_map3(self):
        L1 = np.array([0, 0, 0, 2, 2, 2, 3, 3, 3])
        L2 = np.array([2, 2, 2, 1, 1, 1, 3, 3, 3])
        newL2 = best_map(L1, L2)
        ans = np.array([0, 0, 0, 2, 2, 2, 3, 3, 3])
        self.assertTrue(np.array_equal(newL2, ans))

    def test_best_map4(self):
        L1 = np.array([1, 2, 3, 4])
        L2 = np.array([2, 1, 3, 4])
        newL2 = best_map(L1, L2)
        ans = np.array([1, 2, 3, 4])
        self.assertTrue(np.array_equal(newL2, ans))

if __name__ == '__main__':
    unittest.main()
