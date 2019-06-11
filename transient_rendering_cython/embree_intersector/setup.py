from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

sourcefiles = ['embree_intersector.pyx', 'c_embree_intersector.cpp', 'c_mesh.cpp']
setup(
    ext_modules = cythonize([Extension("embree_intersector", 
                                       sources=sourcefiles, language="c++", libraries=["embree3"])]),

    include_dirs = [np.get_include()]
    )
