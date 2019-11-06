from cmlkit.representation import Representation
from cmlkit.engine import parse_config
from dscribe.descriptors import MBTR as dsMBTR


class MBTR(Representation):
    """MBTR Representation (implemented in DScribe).

    For details, see https://singroup.github.io/dscribe/tutorials/mbtr.html
    or the dscribe source.

    Syntax for parameters is trying to find a mid-way point between
    dscribe and cmlkit.

    Parameters:
        mbtr_1: Inner config dict for k=1 MBTR, or None
        mbtr_2: Inner config dict for k=2 MBTR, or None
        mbtr_3: Inner config dict for k=2 MBTR, or None
        norm: Either None, "l2_each", or "n_atoms"
        normalize_gaussians: Bool
        flatten: Bool, default True
        sparse: Bool, default False (True is untested in cmlkit)



    Each config dict has keys:
        elems: Elements for which we compute MBTR (must be the the same for all)
        start: Value of the first MBTR bin
        stop: Value of last bin
        num: Number of bins
        acc: Float specifying when to stop counting contributions.
        geomf: String specifying the geometry function
            k=1: "atomic_number"
            k=2: "1/distance", "distance"
            k=3: "angle", "cos_angle"
        weightf: String specifying the weight function:
            "unity": No weighting (will diverge in periodic case)
            "exp": Exp(-ls x). Parametrised in MBTR standard way, i.e.
                 {"exp": {"ls": s}}
        broadening: Value of exponential bin broadening.
            (smaller values -> smaller width)

    """

    kind = "ds_mbtr"
    default_context = {"n_jobs": 1, "verbose": False}

    def __init__(
        self,
        mbtr_1=None,
        mbtr_2=None,
        mbtr_3=None,
        normalize_gaussians=True,
        norm=None,
        flatten=True,
        sparse=False,
        context={},
    ):
        super().__init__(context=context)

        self.ds_config = _to_dscribe_config(
            mbtr_1=mbtr_1,
            mbtr_2=mbtr_2,
            mbtr_3=mbtr_3,
            normalize_gaussians=normalize_gaussians,
            norm=norm,
            flatten=flatten,
            sparse=sparse,
        )

        self.config = {
            "mbtr_1": mbtr_1,
            "mbtr_2": mbtr_2,
            "mbtr_3": mbtr_3,
            "normalize_gaussians": normalize_gaussians,
            "norm": norm,
            "sparse": sparse,
        }

    def _get_config(self):
        return self.config

    def compute(self, data):
        if data.b is None:
            ds_mbtr = dsMBTR(**{**self.ds_config, "periodic": False})
        else:
            ds_mbtr = dsMBTR(**{**self.ds_config, "periodic": True})

        rep = ds_mbtr.create(
            data.as_Atoms(),
            n_jobs=self.context["n_jobs"],
            verbose=self.context["verbose"],
        )

        return rep


def _to_dscribe_config(
    mbtr_1=None,
    mbtr_2=None,
    mbtr_3=None,
    normalize_gaussians=True,
    norm=None,
    flatten=True,
    sparse=False,
):
    result = {}
    elems = None
    if mbtr_1 is not None:
        result["k1"] = _to_single_mbtr_config(mbtr_1)
        elems = mbtr_1["elems"]
    if mbtr_2 is not None:
        result["k2"] = _to_single_mbtr_config(mbtr_2)
        if elems is not None:
            assert elems == mbtr_2["elems"]
        else:
            elems = mbtr_2["elems"]
    if mbtr_3 is not None:
        result["k3"] = _to_single_mbtr_config(mbtr_3)
        if elems is not None:
            assert elems == mbtr_3["elems"]
        else:
            elems = mbtr_3["elems"]

    assert len(result) > 0, "At least one MBTR must be specified."
    result["normalization"] = _to_norm(norm)
    result["normalize_gaussians"] = normalize_gaussians
    result["flatten"] = flatten
    result["sparse"] = sparse
    result["species"] = elems

    return result


def _to_single_mbtr_config(config):
    result = {}
    result["grid"] = {
        "min": config["start"],
        "max": config["stop"],
        "n": config["num"],
        "sigma": config["broadening"],
    }
    result["geometry"] = {"function": _to_geomf(config["geomf"])}
    result["weighting"] = _to_weightf(config["weightf"], config["acc"])

    return result


def _to_geomf(geomf):
    mapping = {
        "atomic_number": "atomic_number",
        "1/distance": "inverse_distance",
        "distance": "distance",
        "angle": "angle",
        "cos_angle": "cosine",
    }

    try:
        return mapping[geomf]
    except KeyError:
        raise ValueError(f"Geometry function {geomf} unknown.")


def _to_weightf(weightf, acc):
    if weightf == "unity":
        return {"function": "unity", "cutoff": acc}
    else:
        kind, inner = parse_config(weightf)
        assert kind == "exp"
        return {"function": "exp", "cutoff": acc, "scale": inner["ls"]}


def _to_norm(norm):
    if norm is None:
        return "none"
    else:
        return norm
