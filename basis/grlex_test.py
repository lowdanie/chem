import dataclasses
import unittest

import numpy as np

from basis import grlex


@dataclasses.dataclass
class _GenerateGrlexTriplesTestCase:
    max_degree: int
    expected: np.ndarray


_GENERATE_GRLEX_TRIPLES_TEST_CASES = [
    _GenerateGrlexTriplesTestCase(
        max_degree=0,
        expected=np.array([[0, 0, 0]]),
    ),
    _GenerateGrlexTriplesTestCase(
        max_degree=1,
        expected=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    ),
    _GenerateGrlexTriplesTestCase(
        max_degree=2,
        expected=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [2, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [0, 2, 0],
                [0, 1, 1],
                [0, 0, 2],
            ]
        ),
    ),
]


class GrlexTest(unittest.TestCase):
    def test_generate_grlex_triples(self):
        for case in _GENERATE_GRLEX_TRIPLES_TEST_CASES:
            triples = grlex.generate_grlex_triples(case.max_degree)

            self.assertTrue(
                np.array_equal(triples, case.expected),
                msg=f"Failed for max_degree={case.max_degree}. "
                f"Got {triples}, expected {case.expected}",
            )


if __name__ == "__main__":
    unittest.main()
