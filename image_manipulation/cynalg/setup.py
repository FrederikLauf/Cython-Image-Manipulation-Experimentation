from setuptools import setup
from Cython.Build import cythonize

# python setup.py build_ext -i
setup(ext_modules=cythonize('linalg_pretty.pyx', annotate=True))

