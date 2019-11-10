from unittest import TestCase
import numpy as np

from cmlkit import Dataset

from cscribe.mbtr import LMBTR

# Not really testing for correctness, but testing for not being broken.


class TestLMBTR2(TestCase):
    def setUp(self):
        self.data = Dataset(
            z=np.array([[1, 2, 2]]),
            r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        )

    def test_lmbtr_2(self):
        mbtr_2 = {
            "start": 0,
            "stop": 1,
            "num": 5,
            "geomf": "1/distance",
            "weightf": "unity",
            "broadening": 0.01,
            "acc": 0.001,
        }

        mbtr = LMBTR(
            elems=[1, 2],
            mbtr_2=mbtr_2,
            flatten=True,
            normalize_gaussians=True,
            stratify=False,
        )

        computed = mbtr(self.data)
        print(computed)

        # => [[[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0 4.0]
        #      [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 4.0]
        #      [0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0]]]
        # Each line is 3 blocks of size 5: X-X, X-1, X-2
        # Atom 1 sees two distances with 2, (1/2=0.5 and 1/1=1)
        # Atom 2 sees one each to 1 and 2, (1/1=1)
        # so does Atom 3 (but different distances) (1/1 and 1/2)
        self.assertEqual(len(computed[0][0]), 15)
        self.assertEqual(computed[0][0][14], 4)
        self.assertEqual(computed[0][0][12], 4)
        self.assertEqual(computed[0][1][9], 4)

    def test_lmbtr_2_strat(self):
        mbtr_2 = {
            "start": 0,
            "stop": 1,
            "num": 5,
            "geomf": "1/distance",
            "weightf": "unity",
            "broadening": 0.01,
            "acc": 0.001,
        }

        mbtr = LMBTR(
            elems=[1, 2],
            mbtr_2=mbtr_2,
            flatten=True,
            normalize_gaussians=True,
            stratify=True,
        )

        computed = mbtr(self.data)
        print(computed)

        # =>    [[[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0 4.0 0.0 0.0
        #          0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
        #         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        #          0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 4.0]
        #         [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        #          0.0 0.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0]]]
        # Same as before, but now in 2 blocks.
        self.assertEqual(len(computed[0][0]), 2 * 15)
        self.assertEqual(computed[0][0][14], 4)
        self.assertEqual(computed[0][0][12], 4)
        self.assertEqual(computed[0][1][15+9], 4)
