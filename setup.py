#-*- coding: utf-8 -*-
'''
Script to extract and preprocess data to be used in a debugginator model

Created by: Andrew Younger
2022-01-10
'''

from setuptools import setup, find_packages

with open('src/__init__.py') as f:
    info = {}
    for line in f.readlines():
        if line.startswith('__version__'):
            exec(line,info)
            break

setup(
    name='debugginator',
    version=info['__version__'],
    author='Andrew Younger',
    packages=['src'] + ['src.'+pkg for pkg in find_packages('src')],
    install_requires=['tensorflow>=2.7.0',
                      'pandas>=1.3.4',
                      'numpy>=1.21.4',
                      'scikit-learn>=1.0.1'
                     ]
)