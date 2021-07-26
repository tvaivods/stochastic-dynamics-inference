#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:25:15 2021

@author: tomass
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from dataprep import *
from stochastic_sindy import *
import multiprocessing as mp
from timeit import default_timer as timer
from datetime import timedelta
from tqdm import tqdm
from bases import *

y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

exp_lo = 3          # Shortest trajectory of length 10^exp_lo
exp_hi = 7          # Longest trajectory of length 10^exp_hi
n_lengths = 9      # Number of different lengths in this range
n_strands = 10      # N. of samples per length of trajectory

n_traj = 5          # Trajectories in a sample
x0s = np.linspace(-0.606,4.27,n_traj)
K = 5               # Number of CV folds
diffusion = 1       # Diffusion coefficient
dt = 5e-3           # Timestep length
n_bins = 90

basis = standard_basis
n_features = standard_n_features
basis_name = standard_name

def mp_ts(x0, n, diffusion):      # For multiprocessing of data generation
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)

def gen_stat_err_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, x0s, K,
                           diffusion, dt, n_bins, basis, n_features, basis_name):
    
    suffix = f"_{exp_lo}-{exp_hi}({n_lengths}n{n_strands}s)D{diffusion:.3f}-{basis_name}"
    header = f"Parameters: n_bins={n_bins}, n_traj={n_traj},"\
             + f" diffusion={diffusion}, dt={dt}, basis: {basis_name}"\
             + f"initial positions: {x0s}"
    
    print(f"Generating statistical survival data for {n_lengths} trajectory lengths between "
          + f"{10**exp_lo} and {10**exp_hi}, with {n_strands} strands per trajectory length. "
          + f"\n{header}\n")
    start = timer()
    
    ns = np.int32(10**np.linspace(exp_lo,exp_hi,n_lengths))
    
    error_list = []
    
    for n in ns:                      # for each trajectory length
        print(f"n = {n}")
        
        pbar = tqdm(range(n_strands), leave = True)
        for i in pbar:      # for each strand
            pbar.set_postfix_str(f'Computing strand {i+1}/{n_strands}')
            
            # Generating the data
            x_multiple = []
            # x0s = np.full(n_traj, 2)
            traj_lengths = np.full(n_traj, n)
            diffusions = np.full(n_traj, diffusion)
            pool = mp.Pool()
            mp_arg = np.transpose([x0s, traj_lengths, diffusions])
            x_multiple = pool.starmap(mp_ts, mp_arg)
            pool.close()
            pool.join()
            
            x_multiple = np.array(x_multiple).T
            
            # Computing the matrices
            Y_multiple = ps.differentiation.FiniteDifference(order = 1)._differentiate(
                x_multiple, dt
            )
            Y_single = Y_multiple.reshape(-1, 1, order = 'F')
            x_single = x_multiple.reshape(-1, 1, order = 'F')
            
            # Computing the binned matrices
            x_binned, Y_binned, weights = bin_data(x_single, Y_single, n_bins,
                                               width_type = 'equal')
            basis.fit(x_binned)
            X_binned = basis.transform(x_binned)
            W = np.diag(weights)
            
            # Running the SSR algorithm
            errors = CV_scores(np.matmul(W,X_binned), np.matmul(W,Y_binned), K = K) # delta^2 !!!
            error_list.append(np.concatenate([[n], errors]))
    
    error_list = np.array(error_list)
    
    end = timer()
    print(f"\nDone! Process duration: {timedelta(seconds=end-start)}.")
    
    np.savetxt("stat_analysis/errdata"+suffix+".csv", error_list,
               delimiter = ',', header=header)
    
    print(f"Data saved in:\nstat_analysis/errdata{suffix}.csv")
    
    return suffix

if __name__ == '__main__':
    gen_stat_err_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, K,
                        diffusion, dt, n_bins, basis, n_features, basis_name)