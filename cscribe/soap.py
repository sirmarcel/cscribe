from cmlkit.representation import Representation
from dscribe.descriptors import SOAP as dsSOAP

from .conversion import to_local


class SOAP(Representation):
    """SOAP Representation (implemented in DScribe).

    For details, see https://singroup.github.io/dscribe/tutorials/soap.html
    or the dscribe source.

    Parametrisation here is identical to "original" quippy SOAP,
    with the addition of the "rbf" argument.

    Parameters:
        elems: Elements for which we compute SOAP
        cutoff: Cutoff radius
        sigma: Broadening
        n_max: Number of radial basis functions
        l_max: Number of angular basis functions
        rbf: Radial basis set. Either "gto" for Gaussians,
            or "polynomial" for a polynomial basis set more
            similar to the "original" SOAP approach.

    """

    kind = "ds_soap"
    default_context = {"n_jobs": 1, "verbose": False}

    def __init__(self, elems, cutoff, sigma, n_max, l_max, rbf="gto", context={}):
        super().__init__(context=context)

        self.config = {
            "elems": elems,
            "cutoff": cutoff,
            "sigma": sigma,
            "n_max": n_max,
            "l_max": l_max,
            "rbf": rbf,
        }

    def _get_config(self):
        return self.config

    def compute(self, data):
        if data.b is None:
            ds_soap = dsSOAP(
                species=self.config["elems"],
                rcut=self.config["cutoff"],
                nmax=self.config["n_max"],
                lmax=self.config["l_max"],
                sigma=self.config["sigma"],
                rbf=self.config["rbf"],
                crossover=True,
                periodic=False,
            )

        else:
            ds_soap = dsSOAP(
                species=self.config["elems"],
                rcut=self.config["cutoff"],
                nmax=self.config["n_max"],
                lmax=self.config["l_max"],
                sigma=self.config["sigma"],
                rbf=self.config["rbf"],
                crossover=True,
                periodic=True,
            )

        rep = ds_soap.create(
            data.as_Atoms(),
            n_jobs=self.context["n_jobs"],
            verbose=self.context["verbose"],
        )

        return to_local(data, rep)
