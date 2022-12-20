from setuptools import setup, find_packages

from montecarlolearning import __version__

# Read requirements from requirements.txt
import codecs
import chardet
def get_encoding(file):
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

encoding = get_encoding('requirements.txt')
install_requires = codecs.open('requirements.txt', 'r', encoding=encoding).read().splitlines()

# Package information
setup(
    name='montecarlolearning',
    version=__version__,

    url='https://github.com/da-roth/NeuronalNetworkTensorflowFramework',
    author='Daniel Roth',
    author_email='daniel-roth@posteo.org',

    packages=find_packages(),
    install_requires=install_requires,
)