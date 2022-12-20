from setuptools import setup, find_packages

from montecarlolearning import __version__

setup(
    name='montecarlolearning',
    version=__version__,

    url='https://github.com/da-roth/NeuronalNetworkTensorflowFramework',
    author='Daniel Roth',
    author_email='daniel-roth@posteo.org',

    packages=find_packages(),
)