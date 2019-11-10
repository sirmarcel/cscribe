from unittest import TestCase
import numpy as np

from cmlkit import Dataset

from cscribe.mbtr import MBTR


class TestMBTR1(TestCase):
    def setUp(self):
        self.data = Dataset(z=np.array([[1, 2]]), r=np.array([[[1, 2, 3], [1, 2, 3]]]))

    def test_mbtr_1(self):
        mbtr_1 = {
            "start": 0,
            "stop": 4,
            "num": 5,
            "geomf": "atomic_number",
            "weightf": "unity",
            "broadening": 0.001,
            "acc": 0.001,
        }

        mbtr = MBTR(elems=[0, 1, 2, 3], mbtr_1=mbtr_1, flatten=True)

        computed = mbtr(self.data)

        print(computed)

        # => [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
        # which is in non-flattened form:
        #  [[0., 0., 0., 0., 0.],
        #   [0., 1., 0., 0., 0.],
        #   [0., 0., 1., 0., 0.],
        #   [0., 0., 0., 0., 0.]],
        # which is baffling, but apparently it's still stratified by element.
        # but ok if it works it works

        self.assertEqual(computed[0][6], 1.0)
        self.assertEqual(computed[0][12], 1.0)


class TestMBTR2(TestCase):
    def setUp(self):
        self.data = Dataset(
            z=np.array([[1, 1]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        )

    def test_mbtr_2(self):
        mbtr_2 = {
            "start": 0,
            "stop": 1,
            "num": 3,
            "geomf": "1/distance",
            "weightf": "unity",
            "broadening": 0.0001,
            "acc": 0.001,
        }

        mbtr = MBTR(elems=[1], mbtr_2=mbtr_2, flatten=True, normalize_gaussians=True)

        computed = mbtr(self.data)
        print(computed)

        # => [[0. 2.0 0.]]
        # There is one inverse distance: 0.5, in the middle of
        # the discretisation interval.
        # the weird scaling is an artifact of the weird settings,
        # see https://github.com/SINGROUP/dscribe/issues/32
        self.assertEqual(computed[0][1], 2.0)
        self.assertEqual(computed[0][2], 0.0)

    def test_mbtr_2_with_parametrized_weightf(self):
        # just make sure things don't explode
        mbtr_2 = {
            "start": 0,
            "stop": 1,
            "num": 3,
            "geomf": "1/distance",
            "weightf": {"exp": {"ls": 1.0}},
            "broadening": 0.0001,
            "acc": 0.001,
        }

        mbtr = MBTR(elems=[1], mbtr_2=mbtr_2, flatten=True, normalize_gaussians=True)

        computed = mbtr(self.data)


class TestMBTR3(TestCase):
    def setUp(self):
        self.data = Dataset(
            z=np.array([[1, 1, 1]]),
            r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        )

    def test_mbtr_3(self):
        mbtr_3 = {
            "start": 0,
            "stop": 360,
            "num": 9,
            "geomf": "angle",
            "weightf": "unity",
            "broadening": 0.0001,
            "acc": 0.001,
        }

        mbtr = MBTR(elems=[1], mbtr_3=mbtr_3, flatten=True, normalize_gaussians=True)

        computed = mbtr(self.data)
        print(computed)

        # => [[0.         0.04444445 0.02222222 0.         0.         0.
        #      0.         0.         0.        ]]
        # There are three angles: once 90 degrees,
        # then twice 45 degrees.

        self.assertGreater(computed[0][1], 0.0)
        self.assertGreater(computed[0][2], 0.0)
