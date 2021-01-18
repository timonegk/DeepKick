from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deep_kick'))

VERSION = 0.1

setup(name='deep_kick',
      version=VERSION,
      description='DeepKick',
      author='Timon Engelke',
      author_email='7engelke@informatik.uni-hamburg.de',
      license='',
      zip_safe=False,
      install_requires=[])
