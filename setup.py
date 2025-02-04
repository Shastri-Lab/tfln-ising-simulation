from setuptools import setup, find_packages

setup(
    name="pysing_machine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy',
        'matplotlib',
        'torch',    
        'dimod',
        # dwave stuff
    ],
)