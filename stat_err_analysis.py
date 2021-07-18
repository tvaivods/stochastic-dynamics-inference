#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Further study of the 1D example from L. Boninsegna and C. Clementi,
“Sparse learning of stochastic dynamical equations,”
J. Chem. Phys., vol. 148, p. 241723, 2018, doi: 10.1063/1.5018409.
Investigation of the SSR result statistical accuracy under different
sample sizes.
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
exp_hi = 4          # Longest trajectory of length 10^exp_hi
n_lengths = 3      # Number of different lengths in this range
n_strands = 10      # N. of samples per length of trajectory

n_bins = 90         # Data bin number
K = 5               # Number of CV folds

ns = np.int32(10**np.linspace(exp_lo,exp_hi,n_lengths))  # Lengths of trajectories
n_traj = 5          # Trajectories in a sample
diffusion = 1       # Diffusion coefficient
dt = 5e-3           # Timestep length

def mp_ts(x0):      # For multiprocessing of data generation
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)

# Specifying the basis functions
basis1 = ps.feature_library.polynomial_library.PolynomialLibrary(degree=7)
basis2 = ps.feature_library.fourier_library.FourierLibrary(n_frequencies=6)
basis = basis1 + basis2

suffix = f"_{exp_lo}-{exp_hi}({n_lengths}n{n_strands}s)"
header = f"Parameters: n_bins={n_bins}, K={K}, n_traj={n_traj}, \
diffusion={diffusion}, dt={dt}"

#%% Running the algorithm
# Produces error_list -- an array of shape [n_lengths*n_strands, n_features+1],
# 1st column containing the trajectory length, with the latter containing the
# errors of the strand corresponding to the n. of non-zero entries.

error_list = []

print(f"Running the cross-validated SSR algorithm for variable trajectory length.\n\
Parameters:\ndt = {dt},\ndiffusion = {diffusion},\nntraj = {n_traj}\n\
{n_lengths} different trajectory lengths ranging from {10**exp_lo} \
to {10**exp_hi}.\n\
Data collected in {n_bins} bins. CV performed with {K} folds of data. \n\
Results averaged over {n_strands} runs.")

time.sleep(0.5)
pbar = tqdm(ns, leave = True)

for n in pbar:                      # for each trajectory length
    for i in range(n_strands):      # for each strand
        pbar.set_postfix_str(f'traj_len: {n}, sample: {i+1}/{n_strands}')
        
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
        errors = CV_scores(np.matmul(W,X_binned), np.matmul(W,Y_binned), K = K)
        error_list.append(np.concatenate([[n], errors]))

error_list = np.array(error_list)

np.savetxt("stat_analysis/trajdata"+suffix+".csv", error_list, delimiter = ',', header=header)

print("Done! Data saved as " + "stat_analysis/trajdata"+suffix+".csv")


#%% Visualising the CV scores

# Reading the data and computing mean, std for strand groups
error_list = np.loadtxt("stat_analysis/trajdata"+suffix+".csv", delimiter = ',')
n_lengths = len(np.unique(error_list[:,0]))
n_strands = error_list.shape[0]/n_lengths
mean_list = []
std_list = []
for i in range(n_lengths):
    mean_list.append(error_list[int(i*n_strands):int((i+1)*n_strands), 1:].mean(axis=0))
    std_list.append(error_list[int(i*n_strands):int((i+1)*n_strands), 1:].std(axis=0))
mean_list = np.array(mean_list)
std_list = np.array(std_list)

# Plotting the data
fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [20, 1]}, dpi=300)

cmap = pl.cm.jet
norm = matplotlib.colors.LogNorm(vmin=np.min(error_list[:,0]),
                                 vmax=np.max(error_list[:,0]))
colors = cmap(norm(error_list[:,0]).data)

ax[0].set_title("CV scores for different sample sizes")
ax[0].set_xlabel("Number of non-zero terms")
ax[0].set_ylabel(r"CV score $\delta$")
ax[0].set_xticks(np.arange(1,error_list.shape[1],1))
ax[0].set_xlim([1,error_list.shape[1]-1])

# Strand plot
for i in range(error_list.shape[0]):
    ax[0].semilogy(np.arange(1,error_list.shape[1], 1), error_list[i, 1:], '.-',
                  color=colors[i], alpha = 0.2, markersize = 1)
# Mean plot
for i in range(n_lengths):
    ax[0].semilogy(np.arange(1,error_list.shape[1], 1), mean_list[i], '.-',
              color=colors[int(i*n_strands)], markersize = 2)
    # ax[0].errorbar(np.arange(1,error_list.shape[1], 1), mean_list[i],  yerr = std_list[i],
    #                 color=colors[int(i*n_strands)], markersize = 2, alpha = 0.75)
ax[0].set_yscale("log")

# Colorbar
matplotlib.colorbar.ColorbarBase(ax = ax[1], cmap = cmap, norm = norm, orientation = "vertical")
ax[1].invert_yaxis()
ax[1].set_ylabel("Trajectory length")

fig.savefig("stat_analysis/errors"+suffix+".png")
plt.show()