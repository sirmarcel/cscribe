"""Convert representations for cmlkit"""

import numpy as np


def to_local(data, rep):
    """Convert dscribe-style atomic rep to cmlkit-style atomic rep.

    dscribe returns local descriptors as one flat array of dimension
    n_total_atoms x dim, whereas cmlkit expects a ndarray-list of length
    n_systems, where each entry is an ndarry of dim n_atoms_system x dim,
    i.e. the local representations for each atom in this particular system.

    The translation between these two notations is done via an offset array,
    which keeps track of which entries in the dscribe array belong to which atom.

    Args:
        data: Dataset instance
        rep: ndarray n_total_atoms x dim

    Returns:
        cmlkit-style atomic representation

    """

    counts = data.info["atoms_by_system"]

    offsets = np.zeros(len(counts) + 1, dtype=int)
    offsets[1::] = np.cumsum(counts)

    return np.array(
        [rep[offsets[i] : offsets[i + 1]] for i in range(data.n)], dtype=object
    )


def in_blocks(data, rep, elems=None):
    """Arrange local representation in blocks by element.

    Some representations (ACSF) are returned without taking the central atom type
    into account. This doesn't work with kernel ridge regression, so we re-arrange
    the respresentation into zero-padded element blocks, like so:

    Let (...) be the representation. Let's say we have Z's 1 and 2. Let's say we
    have one molecule with first atom Z1=1, second Z2=1. Let's say we have dim=2.

    The result of this function will be

    [
     (Z1=1) (0, 0),
     (0, 0), (Z2=1)
    ]

    I.e. we will have separate blocks for each central atom, filled with zeros where
    the central atom type is "not in use".

    Args:
        data: Dataset instance
        rep: cmlkit-style atomic representation
        elems: List of elements to take into account,
            if not specified will use the ones given in data.

    Returns:
        cmlkit-style atomic representation

    """
    if elems is None:
        n_elems = data.info["total_elements"]
        elem_idx = {e: i for i, e in enumerate(data.info["elements"])}
    else:
        n_elems = len(elems)
        elem_idx = {e: i for i, e in enumerate(elems)}

    all_new = []

    for i, rep_system in enumerate(rep):
        dim = len(rep_system[0])
        new = np.zeros((len(rep_system), dim * n_elems))

        for j, rep_atom in enumerate(rep_system):
            idx = elem_idx[data.z[i][j]]
            new[j, idx * dim : (idx + 1) * dim] = rep_atom

        all_new.append(new)

    return np.array(all_new, dtype=object)
