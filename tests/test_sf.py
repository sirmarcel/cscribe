import numpy as np
from unittest import TestCase
import unittest.mock
import shutil
import pathlib

from cmlkit import Dataset

from cscribe.sf import SymmetryFunctions


def fc(r, cutoff):
    return 0.5 * (np.cos(np.pi * r / cutoff) + 1)


def rad_sf(r, eta, mu):
    return np.exp(-eta * (r - mu) ** 2)


def ang_sf(ang, r1, r2, r3, lambd, zeta, eta):
    return (
        2.0 ** (1 - zeta)
        * (1 + (lambd * np.cos(ang))) ** zeta
        * np.exp(-eta * ((r1 ** 2) + (r2 ** 2) + (r3 ** 2)))
    )


class TestSymmetryFunctions(TestCase):
    def test_rad_sf(self):

        data = Dataset(
            z=np.array([[1, 1]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        )

        for i in range(5):
            eta = np.random.random()
            mu = np.random.random()
            cutoff = 6.0

            sf = SymmetryFunctions(
                elems=[1], cutoff=cutoff, sfs=[{"rad": {"eta": eta, "mu": mu}}]
            )

            computed = sf(data)

            print(computed.shape)

            np.testing.assert_almost_equal(computed[0][0][0], fc(2.0, cutoff))

            np.testing.assert_almost_equal(
                computed[0][0][1], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
            )

            np.testing.assert_almost_equal(computed[0][1][0], fc(2.0, cutoff))

            np.testing.assert_almost_equal(
                computed[0][1][1], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
            )

    def test_rad_sf_periodic(self):

        data = Dataset(
            z=np.array([[1, 1]]),
            r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            b=np.array([[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]]),
        )

        # just testing that nothing is broken; cutoff is smaller than cell
        for i in range(5):
            eta = np.random.random()
            mu = np.random.random()
            cutoff = 6.0

            sf = SymmetryFunctions(
                elems=[1], cutoff=cutoff, sfs=[{"rad": {"eta": eta, "mu": mu}}]
            )

            computed = sf(data)

            print(computed.shape)

            np.testing.assert_almost_equal(computed[0][0][0], fc(2.0, cutoff))

            np.testing.assert_almost_equal(
                computed[0][0][1], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
            )

            np.testing.assert_almost_equal(computed[0][1][0], fc(2.0, cutoff))

            np.testing.assert_almost_equal(
                computed[0][1][1], rad_sf(2.0, eta, mu) * fc(2.0, cutoff)
            )

    def test_parametrized_sf(self):

        data = Dataset(
            z=np.array([[1, 1]]), r=np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        )

        cutoff = 5.0
        sf = SymmetryFunctions([1], sfs=[{"rad_centered": {"n": 3}}], cutoff=cutoff)

        delta = (cutoff - 1.5) / 2

        computed = sf(data)
        print(computed)

        np.testing.assert_almost_equal(computed[0][0][0], fc(2.0, cutoff))

        np.testing.assert_almost_equal(
            computed[0][0][1],
            rad_sf(2.0, 0.5 / (0.5 + 0 * delta) ** 2, 0.0) * fc(2.0, cutoff),
        )

        np.testing.assert_almost_equal(
            computed[0][0][2],
            rad_sf(2.0, 0.5 / (0.5 + 1 * delta) ** 2, 0.0) * fc(2.0, cutoff),
        )

        np.testing.assert_almost_equal(
            computed[0][0][3],
            rad_sf(2.0, 0.5 / (0.5 + 2 * delta) ** 2, 0.0) * fc(2.0, cutoff),
        )

    def test_ang_sf(self):

        data = Dataset(
            z=np.array([[1, 1, 1]]),
            r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        )

        for i in range(5):
            eta = np.random.random()
            zeta = np.random.random()
            cutoff = 5.0
            lambd = 1.0 + np.random.random() * 0.3

            sf = SymmetryFunctions(
                [1],
                sfs=[{"ang": {"eta": eta, "zeta": zeta, "lambd": lambd}}],
                cutoff=cutoff,
            )

            computed = sf(data)

            print(computed)

            np.testing.assert_almost_equal(
                computed[0][0][0], fc(1.0, cutoff) + fc(1.0, cutoff)
            )

            np.testing.assert_almost_equal(
                computed[0][0][1],
                ang_sf(np.pi / 2, 1.0, 1.0, np.sqrt(2.0), lambd, zeta, eta)
                * fc(1.0, cutoff)
                * fc(1.0, cutoff)
                * fc(np.sqrt(2.0), cutoff),
            )

            np.testing.assert_almost_equal(
                computed[0][1][0], fc(1.0, cutoff) + fc(np.sqrt(2.0), cutoff)
            )

            np.testing.assert_almost_equal(
                computed[0][1][1],
                ang_sf(np.pi / 4, 1.0, 1.0, np.sqrt(2.0), lambd, zeta, eta)
                * fc(1.0, cutoff)
                * fc(1.0, cutoff)
                * fc(np.sqrt(2.0), cutoff),
            )

    def test_rad_sf_multiple_elements(self):

        data = Dataset(
            z=np.array([[1, 2, 3]]),
            r=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        )

        for i in range(5):
            eta = np.random.random()
            mu = np.random.random()
            cutoff = 5.0
            sf = SymmetryFunctions(
                [1, 2, 3], sfs=[{"rad": {"eta": eta, "mu": mu}}], cutoff=cutoff
            )

            computed = sf(data)

            print(computed)
            print(computed.shape)

            # Atom 0

            # Other Z = 1
            np.testing.assert_almost_equal(computed[0][0][0], 0.0)
            np.testing.assert_almost_equal(computed[0][0][1], 0.0)

            # Other Z = 2
            np.testing.assert_almost_equal(computed[0][0][2], fc(1.0, cutoff))
            np.testing.assert_almost_equal(
                computed[0][0][3], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
            )

            # Other Z = 3
            np.testing.assert_almost_equal(computed[0][0][4], fc(1.0, cutoff))
            np.testing.assert_almost_equal(
                computed[0][0][5], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
            )

            # Atom 1

            # Other Z = 1
            np.testing.assert_almost_equal(computed[0][1][6 + 0], fc(1.0, cutoff))
            np.testing.assert_almost_equal(
                computed[0][1][6 + 1], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
            )

            # Other Z = 2
            np.testing.assert_almost_equal(computed[0][1][6 + 2], 0.0)
            np.testing.assert_almost_equal(computed[0][1][6 + 3], 0.0)

            # Other Z = 3
            np.testing.assert_almost_equal(
                computed[0][1][6 + 4], fc(np.sqrt(2.0), cutoff)
            )
            np.testing.assert_almost_equal(
                computed[0][1][6 + 5],
                rad_sf(np.sqrt(2.0), eta, mu) * fc(np.sqrt(2.0), cutoff),
            )

            # Atom 2

            # Other Z = 1
            np.testing.assert_almost_equal(computed[0][2][12 + 0], fc(1.0, cutoff))
            np.testing.assert_almost_equal(
                computed[0][2][12 + 1], rad_sf(1.0, eta, mu) * fc(1.0, cutoff)
            )

            # Other Z = 2
            np.testing.assert_almost_equal(
                computed[0][2][12 + 2], fc(np.sqrt(2.0), cutoff)
            )
            np.testing.assert_almost_equal(
                computed[0][2][12 + 3],
                rad_sf(np.sqrt(2.0), eta, mu) * fc(np.sqrt(2.0), cutoff),
            )

            # Other Z = 3
            np.testing.assert_almost_equal(computed[0][2][12 + 4], 0.0)
            np.testing.assert_almost_equal(computed[0][2][12 + 5], 0.0)
