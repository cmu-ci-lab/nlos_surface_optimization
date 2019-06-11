from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

sourcefiles = ['el_topo_api.pyx', 'c_el_topo_api.cpp']
setup(
    ext_modules = cythonize([Extension("el_topo_api", 
                                       sources=sourcefiles, language="c++", libraries=["lapack", "blas"],  extra_objects=["/usr/local/eltopo/eltopo3d/libeltopo_release.a"])]),
    include_dirs = [np.get_include()]
    )
