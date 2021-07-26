#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:57:58 2021

@author: tomass
"""

from stat_err_data import *
from termcolor import colored

y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

exp_lo = 3          # Shortest trajectory of length 10^exp_lo
exp_hi = 7          # Longest trajectory of length 10^exp_hi
n_lengths = 9      # Number of different lengths in this range
n_strands = 20      # N. of samples per length of trajectory

n_traj = 5          # Trajectories in a sample
x0s = np.linspace(-0.606,4.27,n_traj)
K = 5
dt = 5e-3           # Timestep length
n_bins = 90

basis = standard_basis
n_features = standard_n_features
basis_name = standard_name

Ds = np.linspace(0,1,5)

print(f"Statistical error data generator for diffusion coefficient values D = ",
      Ds, '\n')

for diffusion in Ds:
    print(colored(f"#########  Generating data for D = {diffusion}.  #########",
                  "blue", "on_white")+"\n")
    gen_stat_err_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, x0s, K,
                      diffusion, dt, n_bins, basis, n_features, basis_name)