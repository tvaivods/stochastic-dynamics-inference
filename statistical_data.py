#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:35:08 2021

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
from datetime import timedelta, datetime
from tqdm import tqdm
from termcolor import colored
from bases import *


y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

exp_lo = 3          # Shortest trajectory of length 10^exp_lo
exp_hi = 5          # Longest trajectory of length 10^exp_hi
n_lengths = 9      # Number of different lengths in this range
n_strands = 10      # N. of samples per length of trajectory

n_traj = 5          # Trajectories in a sample
x0s = np.linspace(-0.606,4.27,n_traj)
K = 0
diffusion = 1       # Diffusion coefficient
noise = 0           # Measurement noise coefficient
dt = 5e-3           # Timestep length
n_bins = 90

basis = standard_basis
n_features = standard_n_features
basis_name = standard_name

def mp_ts(x0, n, diffusion):      # For multiprocessing of data generation
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)

def gen_statistical_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, x0s,
                         K, diffusion, noise, dt, n_bins, basis, n_features,
                           basis_name):
    
    suffix = f"_{exp_lo}-{exp_hi}({n_lengths}n{n_strands}s)"\
                +f"D{diffusion:.3f}N{noise:.3f}-{basis_name}"
    header = f"Parameters: n_bins={n_bins}, n_traj={n_traj}, "\
             + f"diffusion={diffusion}, noise={noise}, dt={dt}, basis: {basis_name}, "\
             + f"initial positions: {x0s}, (K-fold number = {K})"
    
    print(f"Generating statistical data for {n_lengths} trajectory lengths between "
          + f"{10**exp_lo} and {10**exp_hi}, with {n_strands} strands per trajectory length. "
          + f"\n{header}\n")
    start = timer()
    
    ns = np.int32(10**np.linspace(exp_lo,exp_hi,n_lengths))
    time_estim_pos = np.where(ns >= 10000)[0][0]  # Length at which total time is estimated

    mask_lists = []
    order_lists = []
    error_list = []
    
    for n in ns:
        
        mask_list = []
        order_list = []
        
        print(f"n = {n} (#{np.where(n == ns)[0][0]+1}/{len(ns)})")
        
        pbar = tqdm(range(n_strands), leave = True)
        estim_start = timer()
        
        for i in pbar:
            pbar.set_postfix_str(f'Computing strand {i+1}/{n_strands}')
            
            # Generating the data
            x_multiple = []
            x0s = np.full(n_traj, 2)
            traj_lengths = np.full(n_traj, n)
            diffusions = np.full(n_traj, diffusion)
            pool = mp.Pool()
            mp_arg = np.transpose([x0s, traj_lengths, diffusions])
            x_multiple = pool.starmap(mp_ts, mp_arg)
            pool.close()
            pool.join()
            
            x_multiple = np.array(x_multiple).T
            
            # Adding measurement noise
            x_multiple = add_noise(x_multiple, noise)
            
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
            
            # Running the SSR algorithm (for survival matrices and orders)
            _, masks, errors = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned))
            
            opt_n = opt_term_n(errors)
            opt_mask = masks.squeeze()[opt_n-1]
            mask_list.append(opt_mask)
            
            order = survival_to_order(masks)
            order_list.append(order)
            
            # Running the CV SSR algorithm (for errors/CV-scores)
            if K>0:
                errors = CV_scores(np.matmul(W,X_binned), np.matmul(W,Y_binned), K = K) # delta^2 !!!
                error_list.append(np.concatenate([[n], errors]))
            
            estim_end = timer()
            time_per_pt = (estim_end-estim_start)/n/n_traj/n_strands
            pbar.set_postfix_str(f'{time_per_pt:0.3e} s/pt')
        
        mask_list = np.array(mask_list)
        order_list = np.array(order_list)
        
        mask_lists.append(mask_list)
        order_lists.append(order_list)
        
        if (ns[time_estim_pos] == n and len(ns)>1):
            time_left = (ns[time_estim_pos+1:]).sum()*time_per_pt*n_traj*n_strands
            t_finish = (timedelta(seconds = time_left)+datetime.now()).strftime("%H:%M")
            print(colored(f"Estimated completion time: {t_finish}", "red"))
    
    mask_lists = np.array(mask_lists)
    order_lists = np.array(order_lists)
    error_list = np.array(error_list)
    
    end = timer()
    print(f"\nDone! Process duration: {timedelta(seconds=end-start)}.")
    
    mask_table = np.zeros([n_lengths*n_strands, n_features+1])
    mask_table[:,1:] = mask_lists.reshape(-1,n_features)
    mask_table[:,0] = np.array([[n]*n_strands for n in ns]).reshape(1,-1)
    mask_table = mask_table.astype('int32')
    np.savetxt("stat_analysis/survmasks"+suffix+".csv", mask_table,
               delimiter = ',', header=header)
    
    order_table = np.zeros([n_lengths*n_strands, n_features+1])
    order_table[:,1:] = order_lists.reshape(-1,n_features)
    order_table[:,0] = np.array([[n]*n_strands for n in ns]).reshape(1,-1)
    order_table = order_table.astype('int32')
    np.savetxt("stat_analysis/survorders"+suffix+".csv", order_table,
               delimiter = ',', header=header)
    
    if K>0:
        np.savetxt("stat_analysis/errdata"+suffix+".csv", error_list,
                   delimiter = ',', header=header)
    
    print(colored(f"Data saved in:\nstat_analysis/survmasks{suffix}.csv"
          +f"\nstat_analysis/survorders{suffix}.csv", "blue"))
    if K>0:
        print(colored(f"stat_analysis/errdata{suffix}.csv", "blue"))
    
    return suffix

if __name__ == "__main__":
    gen_statistical_data(y, exp_lo, exp_hi, n_lengths, n_strands, n_traj, x0s, K,
                        diffusion, noise, dt, n_bins, basis, n_features, basis_name)