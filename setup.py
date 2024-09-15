from setuptools import setup, find_packages

setup(
    name='sionna-changen',
    version='0.1.0',
    description='Channel generative modeling.',
    packages=find_packages(),
    install_requires=[
        'jupyterlab-widgets~=3.0.12',
    	'sionna',
    	'mitsuba',
    	'drjit',
        'torch',
        'pandas',
        'numpy',
        'matplotlib',
        'geopy'
    ],
    scripts=["scripts/data_generation.py"]
)
