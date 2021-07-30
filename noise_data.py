#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:22:50 2021

@author: tomass
"""

from statistical_data import *

noise_array = np.linspace(0,0.2,9)

y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

exp_lo = 6          # Shortest trajectory of length 10^exp_lo
exp_hi = 6          # Longest trajectory of length 10^exp_hi
n_lengths = 1      # Number of different lengths in this range
n_strands = 5      # N. of samples per length of trajectory

n_traj = 5          # Trajectories in a sample
x0s = np.linspace(-0.606,4.27,n_traj)
K = 0
diffusion = 1       # Diffusion coefficient
dt = 5e-3           # Timestep length
n_bins = 90

basis = standard_basis
n_features = standard_n_features
basis_name = standard_name

print(f"Statistical data generator for noise values N = ",
      noise_array, '\n')

for noise in noise_array:
    gen_statistical_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, x0s,
                         K, diffusion, noise, dt, n_bins, basis, n_features,
                         basis_name)