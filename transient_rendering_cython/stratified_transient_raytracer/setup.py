from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

sourcefiles = ['rng_sse.cpp', 'sampler.cpp', 'stratifiedTransientRenderer.cpp', 'stratifiedStreamedTransientRenderer.cpp', 'stratifiedStreamedGradientRenderer.cpp', 'renderer.pyx' ]

setup(
    ext_modules = cythonize([Extension("renderer", 
                                       sources=sourcefiles, language="c++", libraries=["embree3"])])
    )
