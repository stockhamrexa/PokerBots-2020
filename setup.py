"""
Used to compile the file equity_calc.pyx into C code.
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("equity_calc.pyx"),
)