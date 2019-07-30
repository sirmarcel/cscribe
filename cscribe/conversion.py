"""Convert representations for cmlkit"""

import numpy as np


def to_local(data, rep):
    """Indices of local descriptors belonging to structures

    dscribe returns local descriptors as one flat array of dimension
    n_total_atoms x dim, whereas cmlkit expects a ndarray-list of length
    n_systems, where each entry is an ndarry of dim n_atoms_system x dim,
    i.e. the local representations for each atom in this particular system.

    The translation between these two notations is done via an offset array,
    which keeps track of which entries in the dscribe array belong to which atom.
    """

    counts = data.info["atoms_by_system"]

    offsets = np.zeros(len(counts) + 1, dtype=int)
    offsets[1::] = np.cumsum(counts)

    return np.array(
        [rep[offsets[i]: offsets[i + 1]] for i in range(data.n)], dtype=object
    )
