# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:19:54 2024

@author: Klest Dedja

"""
import os

# app/bellatrex/__init__.py
from .LocalMethod_class import BellatrexExplain

# expose these functions to outer layers:
__all__ = ['BellatrexExplain']


version_file = os.path.join(os.path.dirname(__file__), 'version.txt')

with open(version_file) as vf:
    __version__ = vf.read().strip()

