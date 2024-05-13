#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['biothings_client>=0.2.6',
'joblib>=1.2.0', 'matplotlib>=3.6.2', 'numpy>=1.23.5','pandas>=1.5.2',
'scipy>=1.10.1', 'seaborn>=0.13.2']

test_requirements = [ ]

setup(
    author="Bharath Ananthasubramaniam",
    author_email='bharath.ananthasubramaniam@hu-berlin.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="The COFE package implements nonlinear dimensionality reduction with a circular constraint on the (dependent) principal components.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='COFE',
    name='COFE',
    packages=find_packages(include=['COFE', 'COFE.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/bharathananth/COFE',
    version='0.1.0',
    zip_safe=False,
)
