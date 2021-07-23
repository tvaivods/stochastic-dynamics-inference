#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:05:20 2021

@author: tomass
"""

import pysindy as ps
import numpy as np

#%% First basis from Boninsegna(2018)

boninsegna1_name = "boninsegna1"

boninsegna1_fns = [
    lambda x : 1,
    lambda x : x,
    lambda x : x**2,
    lambda x : x**3,
    lambda x : x**4,
    lambda x : x**5,
    lambda x : x**6,
    lambda x : x**7,
    lambda x : x**8,
    lambda x : x**9,
    lambda x : x**10,
    lambda x : np.sin(x),
    lambda x : np.cos(x),
    lambda x : np.sin(6*x),
    lambda x : np.cos(6*x),
    lambda x : np.sin(11*x),
    lambda x : np.cos(11*x),
    lambda x : np.tanh(10*x),
    lambda x : -10*np.tanh(10*x)**2 + 10*np.exp(-50*x**2)
]
boninsegna1_names = [
    lambda x : '1',
    lambda x : x,
    lambda x : x+'^2',
    lambda x : x+'^3',
    lambda x : x+'^4',
    lambda x : x+'^5',
    lambda x : x+'^6',
    lambda x : x+'^7',
    lambda x : x+'^8',
    lambda x : x+'^9',
    lambda x : x+'^10',
    lambda x : 'sin('+x+')',
    lambda x : 'cos('+x+')',
    lambda x : 'sin(6*'+x+')',
    lambda x : 'cos(6*'+x+')',
    lambda x : 'sin(11*'+x+')',
    lambda x : 'cos(11*'+x+')',
    lambda x : 'tanh(10*'+x+')',
    lambda x : '-10 tanh(10*'+x+'$)^2$ + 10 exp(-50*'+x+'$**2$)'
]

boninsegna1_basis = ps.CustomLibrary(
    library_functions=boninsegna1_fns, function_names=boninsegna1_names
)
boninsegna1_n_features = 19

#%%

standard_name = "standard"

standard_fns = [
    lambda x : 1,
    lambda x : x,
    lambda x : x**2,
    lambda x : x**3,
    lambda x : x**4,
    lambda x : x**5,
    lambda x : x**6,
    lambda x : x**7,
    lambda x : x**8,
    lambda x : np.sin(x),
    lambda x : np.cos(x),
    lambda x : np.sin(x/2),
    lambda x : np.cos(x/2),
    lambda x : np.sin(2*x),
    lambda x : np.cos(2*x),
    lambda x : np.exp(-2*(x-1)**2),
    lambda x : np.exp(-2*(x-3)**2),
    lambda x : np.sinh(x),
    lambda x : np.cosh(x),
    lambda x : np.tanh(x-2)
]
standard_names = [
    lambda x : '1',
    lambda x : x,
    lambda x : x+'^2',
    lambda x : x+'^3',
    lambda x : x+'^4',
    lambda x : x+'^5',
    lambda x : x+'^6',
    lambda x : x+'^7',
    lambda x : x+'^8',
    lambda x : 'sin('+x+')',
    lambda x : 'cos('+x+')',
    lambda x : 'sin('+x+'/2)',
    lambda x : 'cos('+x+'/2)',
    lambda x : 'sin(2*'+x+')',
    lambda x : 'cos(2*'+x+')',
    lambda x : 'exp(-2*('+x+'-1)^2)',
    lambda x : 'exp(-2*('+x+'-3)^2)',
    lambda x : 'sinh('+x+')',
    lambda x : 'cosh('+x+')',
    lambda x : 'tanh('+x+')'
]

standard_basis = ps.CustomLibrary(
    library_functions=standard_fns, function_names=standard_names
)
standard_n_features = 20