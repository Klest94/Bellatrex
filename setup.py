# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:21:19 2024

@author: Klest Dedja
"""
import os #TODO: consider moving to pathlib for cross-OS consistency.
from setuptools import find_packages, setup

readme_path = os.path.join('README.md')

with open(readme_path, 'r') as f:
    text_description = f.read()


version_file_path = os.path.join('app', 'bellatrex', 'version.txt')

with open(version_file_path) as version_file:
    synch_version = version_file.read().strip()

setup(
    name='bellatrex',
    version=synch_version,
    description='A toolbox for Building Explanations through a LocaLly AccuraTe Rule EXtractor',
    package_dir={'': 'app'}, #where to find the code: in the app subfolder
    packages=find_packages(where='app'),
    include_package_data=True, #include additional files (which ones precisely is determined by the MANIFEST.in file)
    package_data={
        'bellatrex.datasets': ['*.csv'],
    },
    long_description=text_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Klest94/Bellatrex',
    author='Klest Dedja',
    author_email='daneel.olivaw94@gmail.com',
    licence='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',    ],
    install_requires=['scikit-learn >= 1.2',
                      'threadpoolctl>=3.1',
                      'scikit-survival>=0.22',
                      'scipy>=1.11',
                      'pandas>=1.5', #to load tutorial datasets
                      'matplotlib>=3.7'],
    extras_require={
        'dev': ['pytest', 'twine'],
        'gui': ['dearpygui>=1.6.2', 'dearpygui-ext>=0.9.5'] # for Graphical User Interface (demo)
    },
    python_requires='>=3.9, <4',
)