from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

sourcefiles = ['cgal_api.pyx', 'c_cgal_api.cpp']
setup(
    ext_modules = cythonize([Extension("cgal_api", 
                                       sources=sourcefiles, language="c++", libraries=["gmp", "CGAL"])]),
    include_dirs = [np.get_include()]
    )
