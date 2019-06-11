from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

sourcefiles = ['convolution_mkl.cpp', 'rng_sse.cpp', 'sampler.cpp', 'transient_and_gradient.cpp', 'stratifiedStreamedTransientRenderer.cpp', 'stratifiedStreamedGradientRenderer.cpp', 'renderer.pyx' ]

setup(
    ext_modules = cythonize([Extension("renderer", 
                                       sources=sourcefiles, language="c++", libraries=["embree3"])]),
    include_dirs = [np.get_include()]
    )
