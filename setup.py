from setuptools import Extension, setup

setup(ext_modules=[Extension("xnumpy.jitcall", sources=["xnumpy/jitcall.c"])])
