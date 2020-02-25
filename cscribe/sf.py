import numpy as np

from cmlkit.representation import Representation
from cmlkit.representation.sf.config import prepare_config
from cmlkit.engine import parse_config

from dscribe.descriptors import ACSF

from .conversion import to_local, in_blocks


class SymmetryFunctions(Representation):
    """Atom-Centered Symmetry Functions (with DScribe).

    Symmetry functions are an atomic representation,
    consisting of an array of function values computed
    with different parametrisations.

    They were introduced by Jörg Behler in
    Behler, JCP 134, 074106 (2011).

    At the moment, we support only the "standard"
    angular and radial symmetry functions, which are:

    "rad": G_i^2 (eq. 6) in the paper.
    Parameters:
        eta: Width of the Gaussian.
        mu: Shift of center. (R_S in equation.)

    "ang": G_i^4 (eq. 8) in the paper.
    Parameters:
        eta: Width of Gaussian.
        zeta: Roughly, the width of the angular distribution.
        lambd: "Direction" of angular distribution. (Only +-1.)
            (not that `lambda` is a keyword, so it's lambd)

    ***

    Individual symmetry functions are specified in the default
    `cmlkit` config format, i.e. one radial symmetry function is

    `{"rad": {"eta": 1.0, "mu": 0.0}}`.

    In the `dscribe` interface, only a global cutoff can be specified,
    and all symmetry functions are applied to all element combinations.

    Note that the G_i^1 SF is automatically added by `dscribe`.

    ***

    Additionally, parametrization schemes are available for the radial
    symmetry functions. They are:
        shifted: With argument "n", the n symmetry functions
            are distributed so their mean is on a regular grid between 0 and cutoff.
        centered: Same as shifted, but all SFs are centered on 0, with
            widths on a grid up to the cutoff.

    Both schemes are also specified as configs, for instance:

    `{"shifted": {"n": 10}}`

    ***

    Parametrisation schemes can be combined, and augmented with hand-defined SFs.

    Parameters:
        elems: List of elements for which SFs are supposed to be computed
        cutoff: Cutoff
        sfs: List of configs of SFs or configs of parametrization schemes
        stratify: Whether to arrange output in separate blocks depending on
            central element type, default True

    """

    kind = "ds_sf"
    default_context = {"verbose": False, "n_jobs": 1}

    def __init__(self, elems, cutoff, sfs=[], stratify=True, context={}):
        super().__init__(context=context)

        sfs_with_cutoff = []
        for sf in sfs:
            kind, inner = parse_config(sf)
            inner["cutoff"] = cutoff
            sfs_with_cutoff.append({kind: inner})

        self.runner_config = prepare_config(
            elems=elems, elemental=[], universal=sfs_with_cutoff
        )
        self.config = {"elems": elems, "sfs": sfs, "cutoff": cutoff, "stratify": stratify}

    def compute(self, data):
        return compute_symmfs(
            data,
            elems=self.config["elems"],
            cutoff=self.config["cutoff"],
            sfs=self.runner_config["universal"],
            n_jobs=self.context["n_jobs"],
            verbose=self.context["verbose"],
            stratify=self.config["stratify"],
        )

    def _get_config(self):
        return self.config


def compute_symmfs(data, elems, cutoff, sfs, stratify=True, n_jobs=1, verbose=False):
    g2_params, g4_params = make_params(sfs)

    periodic = data.b is not None

    acsf = ACSF(
        rcut=cutoff,
        g2_params=g2_params,
        g4_params=g4_params,
        species=elems,
        sparse=False,
        periodic=periodic,
    )

    rep = acsf.create(data.as_Atoms(), n_jobs=n_jobs, verbose=verbose)

    if stratify:
        return in_blocks(data, to_local(data, rep), elems=elems)
    else:
        return to_local(data, rep)


def make_params(sfs):
    g2_params = []
    g4_params = []

    for sf in sfs:
        kind, inner = parse_config(sf)

        if kind == "rad":
            g2_params.append([inner["eta"], inner["mu"]])
        elif kind == "ang":
            g4_params.append([inner["eta"], inner["zeta"], inner["lambd"]])
        else:
            raise ValueError(
                f"ACSF kind {kind} is not yet implemented. (Allowed: rad and ang.)"
            )

    if len(g2_params) == 0:
        g2_params = None
    else:
        g2_params = np.array(g2_params)

    if len(g4_params) == 0:
        g4_params = None
    else:
        g4_params = np.array(g4_params)

    return g2_params, g4_params
