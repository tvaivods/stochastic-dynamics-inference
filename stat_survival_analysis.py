#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Further study of the 1D example from L. Boninsegna and C. Clementi,
“Sparse learning of stochastic dynamical equations,”
J. Chem. Phys., vol. 148, p. 241723, 2018, doi: 10.1063/1.5018409.
Investigation of discrepancies between optimal functions and the true
functions.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from dataprep import *
from stochastic_sindy import *
import multiprocessing as mp
import time
from tqdm import tqdm

#%% Setting up parameters and prerequisites

# Force function
y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

exp_lo = 3          # Shortest trajectory of length 10^exp_lo
exp_hi = 6          # Longest trajectory of length 10^exp_hi
n_lengths = 10      # Number of different lengths in this range
n_strands = 15      # N. of samples per length of trajectory
ns = np.int32(10**np.linspace(exp_lo,exp_hi,n_lengths))

n_traj = 5          # Trajectories in a sample
diffusion = 1       # Diffusion coefficient
dt = 5e-3           # Timestep length

def mp_ts(x0):      # For multiprocessing of data generation
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)

#%% Specifying the basis functions
# basis1 = ps.feature_library.polynomial_library.PolynomialLibrary(degree=7)
# basis2 = ps.feature_library.fourier_library.FourierLibrary(n_frequencies=6)
# basis = basis1 + basis2
basis1_fns = [
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
basis1_names = [
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

basis = ps.CustomLibrary(
    library_functions=basis1_fns, function_names=basis1_names
)
n_features = 19

# Defining the true mask
true_mask = np.zeros(n_features, dtype = bool)
true_mask[0:4] = True

#%% Compute an array of optimal masks and orders

mask_lists = []
order_lists = []

for n in ns:

    mask_list = []
    order_list = []
    
    print(f"n = {n}")
    
    pbar = tqdm(range(n_strands), leave = True)
    
    for i in pbar:
        pbar.set_postfix_str(f'Computing strand {i+1}/{n_strands}')
        
        # Generating the data
        x_multiple = []
        x0s = np.full(n_traj, 2)
        pool = mp.Pool()
        x_multiple = pool.map(mp_ts, x0s)
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
        _, masks, errors = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned))
        
        opt_n = opt_term_n(errors)
        opt_mask = masks.squeeze()[opt_n-1]
        mask_list.append(opt_mask)
        
        order = survival_to_order(masks)
        order_list.append(order)
    
    mask_list = np.array(mask_list)
    order_list = np.array(order_list)
    
    mask_lists.append(mask_list)
    order_lists.append(order_list)

mask_lists = np.array(mask_lists)
order_lists = np.array(order_lists)

#%% Survival matrix plot
order_mean = np.mean(order_list, axis = 0)
order_std = np.std(order_list, axis = 0)

sorted_pos = np.argsort(order_mean)
mean_order = np.zeros(len(sorted_pos), dtype = int)
for i in range(len(sorted_pos)):
    mean_order[sorted_pos[i]] = i+1
survival_matrix = order_to_survival(mean_order)

fig = plt.figure(figsize = (6,6), dpi = 200)

extent = [.5,n_features+.5,n_features+.5,.5]
im = plt.imshow(survival_matrix, cmap = 'Blues', extent = extent)

plt.axhline(y=opt_n, color = 'cyan', linestyle = '--')

plt.xticks(np.arange(1,n_features+1,1))
plt.yticks(np.arange(1,n_features+1,1))
plt.xlabel("Basis function index")
plt.ylabel("n")

plt.errorbar(np.arange(1,n_features+1,1), order_mean, order_std, fmt = 'r.',
             markersize = 8, capsize = 4)

plt.ylim([n_features+0.5,0.5])

#%% Hamming distances

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    ham_dist = np.count_nonzero(mask_lists[i] != true_mask, axis = 1)
    avg_ham_dist = np.mean(ham_dist)
    std_ham_dist = np.std(ham_dist)
    plt.errorbar(ns[i], avg_ham_dist, std_ham_dist, fmt = 'k.', markersize = 8, capsize = 4)
plt.xscale("log")
plt.ylim([0, 5])
plt.ylabel(f"Mean Hamming distance (avg. over {n_strands} samples)")
plt.xlabel("Trajectory length")
plt.title("Hamming distance between true and optimal function patterns")
plt.grid()

#%% Jaccard distances

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    jac_coeff = np.count_nonzero(mask_lists[i] & true_mask, axis = 1)/\
        np.count_nonzero(mask_lists[i] | true_mask, axis = 1)
    avg_jac_dist = np.mean(1-jac_coeff)
    std_jac_dist = np.std(jac_dist)
    plt.errorbar(ns[i], avg_jac_dist, std_jac_dist, fmt = 'k.', markersize = 8, capsize = 4)
plt.xscale("log")
plt.ylabel(f"Mean Jaccard distance (avg. over {n_strands} samples)")
plt.xlabel("Trajectory length")
plt.title("Jaccard distance between true and optimal function patterns")
plt.grid()

#%% Number of true and false functions in optimal solution

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    n_true = np.count_nonzero(mask_lists[i,:,0:3], axis = 1)
    n_false = np.count_nonzero(mask_lists[i,:,4:], axis = 1)
    n_terms = np.count_nonzero(mask_lists[i], axis = 1)
    
    plt.errorbar(ns[i], np.mean(n_terms), np.std(n_terms), fmt = 'k.',
                 alpha = 0.8, markersize = 8, capsize = 4)
    plt.errorbar(ns[i], np.mean(n_true), np.std(n_true), fmt = 'g.',
                 alpha = 0.8, markersize = 8, capsize = 4)
    plt.errorbar(ns[i], np.mean(n_false), np.std(n_false), fmt = 'r.',
                 alpha = 0.8, markersize = 8, capsize = 4)
plt.xscale("log")
plt.ylim([0, 5])
plt.ylabel(f"Mean number of functions (avg. over {n_strands} samples)")
plt.xlabel("Trajectory length")
plt.title("Optimal solution function number")
plt.legend(["N. of functions in solution",
            "N. of correct functions",
            "N. of incorrect functions"],)
plt.grid()
plt.axhline(y=4, color = 'grey', linestyle = '--')
plt.axhline(y=0, color = 'grey', linestyle = '--')

#%%

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    n_match = np.count_nonzero(mask_lists[i] & true_mask, axis = 1)
    n_diff = np.count_nonzero(mask_lists[i] & ~true_mask, axis = 1)
    
    plt.errorbar(ns[i], np.mean(n_match), np.std(n_match), fmt = 'g.',
                  alpha = 0.8, markersize = 8, capsize = 4)
    plt.errorbar(ns[i], np.mean(n_diff), np.std(n_diff), fmt = 'r.',
                  alpha = 0.8, markersize = 8, capsize = 4)
plt.xscale("log")
plt.ylim([0, 5])
plt.ylabel(f"Mean number of functions (avg. over {n_strands} samples)")
plt.xlabel("Trajectory length")
plt.title("N. of matching/differing terms between the optimal and true solutions")
plt.legend(["N. of matching functions",
            "N. of differing functions"],)
plt.grid()
plt.axhline(y=4, color = 'grey', linestyle = '--')
plt.axhline(y=0, color = 'grey', linestyle = '--')

#%%

fig, ax = plt.subplots(1,2,figsize = (6,5),
                       gridspec_kw={'width_ratios': [20, 1]}, dpi=200)

cmap = pl.cm.winter
norm = matplotlib.colors.LogNorm(vmin=ns[0],
                                 vmax=ns[-1])
colors = cmap(norm(ns))

for i in range(len(ns)):
    n_match = np.count_nonzero(mask_lists[i] & true_mask, axis = 1)
    n_miss = np.count_nonzero(mask_lists[i] & ~true_mask, axis = 1)
    
    ax[0].errorbar(np.mean(n_match), np.mean(n_miss), np.std(n_match), np.std(n_miss),
                 alpha = 0.8, markersize = 8, capsize = 4, color = colors[i])
ax[0].scatter(4,0, color = 'k', s = 40)
ax[0].set_ylabel("N. of incorrect functions")
ax[0].set_xlabel("N. of correct functions")
ax[0].set_title("Optimal solution function number")
ax[0].grid()

matplotlib.colorbar.ColorbarBase(ax = ax[1], cmap = cmap, norm = norm, orientation = "vertical")
ax[1].invert_yaxis()
ax[1].set_ylabel("Trajectory length")