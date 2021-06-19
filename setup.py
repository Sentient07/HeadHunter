### Head detection

import io
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with io.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Handle version number with optional .dev postfix when building a develop branch
# on AppVeyor.
VERSION = '1.1.0'

setup(
    name='head_detection',
    version=VERSION,
    description='Head detector for plug and predict',
    author='Ramana Subramanyam',
    url='https://gitlab.inria.fr/rsundara/head_detection',
    license='MIT',
    packages=['head_detection.models', 'head_detection.vision', 'head_detection.data'],
    include_package_data=True,
    keywords='head detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
