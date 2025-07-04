from setuptools import setup
from Cython.Build import cythonize

# python setup.py build_ext -i
setup(ext_modules=cythonize(['linalg.pyx', 'cylantro.pyx'], annotate=True))

