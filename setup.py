import os
from distutils.core import setup
from setuptools import find_packages


def read(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    contents = open(path).read()
    return contents


setup(name='rrts',
      version='1.0.0',
      packages=find_packages(where='./'),
      description='RRTs Planners',
      long_description=read('README.rst'),
      author='Gabel Liemann',
      author_email='troubleli233@gmail.com',
      url='https://github.com/liespace/pyRRTs',
      license='MIT',
      install_requires=[
          'numpy',
          'numba',
          'reeds_shepp',
          'matplotlib',
          'cv2',
          'scipy']
      )
