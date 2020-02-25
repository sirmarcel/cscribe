# cscribe 🐫🖋️

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cscribe.svg) [![PyPI](https://img.shields.io/pypi/v/cscribe.svg)](https://pypi.org/project/cscribe/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) 

`cscribe` is a [`cmlkit`](https://github.com/sirmarcel/cmlkit/) plugin for [`dscribe`](https://github.com/SINGROUP/dscribe).

It provides `cmlkit`-style `Components` for the representations implemented in `dscribe`. At the moment, it supports:

- `SOAP`: Supported, tested, used in [`repbench`](https://marcel.science/repbench).
- `SF`: Supported, but only `g2` and `g4`. Untested in production.
- `MBTR`: Supported, but untested in production. Local MBTR is also supported, but also untested.
- Coulomb matrix, sine matrix, or ewald sum matrix are not currently supported. (Please submit a pull request!)

In general, `cscribe` implements a subset of the full capabilities of `dscribe`, in order to stay consistent with the choices made in `cmlkit`. For instance, you can't specify whether `periodic` is turned on or not, and `sparse` is not implemented. Please feel free to build your own customised `Components` based on the code here!

The exact parameters are documented in the code itself, please have a look!

## Installation

```
pip install cscribe
export CML_PLUGINS=cscribe
````

Or add `cscribe` to `$CML_PLUGINS` in your `.bashrc` or equivalent.
