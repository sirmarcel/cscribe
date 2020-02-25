## Technical Notes

This is a very straightforward package: `conversion.py` implements a few common functions needed to translate from `dscribe`'s representation format to the one used in `cmlkit`, and the other `.py` files contain the interfaces for the different representations.

Parameters are defined in the class docstrings of each `Component`.

Please note that this is very early stage software. Help make it more tested!

### Development practice

In true "science software" fashion, there is no real practice here. Development currently happens on the main branch. Release versions are not supposed to be broken (i.e. the tests should pass) and are tagged. Tests just use `unittest`, I recommend `nose` as test runner.

There are no plans to add generated documentation. Docstrings are supposed to be formatted Google style. Code is formatted with `black`.
