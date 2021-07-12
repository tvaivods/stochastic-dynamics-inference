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

y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3

npow_lo = 3
npow_hi = 4
nnum = 10
bin_n = 90
K = 5   # number of CV folds

ns = np.int32(10**np.linspace(npow_lo,npow_hi,nnum))  # Lengths of trajectories
n_traj = 5                              # Trajectories in a sample
diffusion = 1
dt = 5e-3

n_sample = 10    # n. of samples per length of trajectory

def mp_ts(x0):
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)

basis1 = ps.feature_library.polynomial_library.PolynomialLibrary(degree=7)
basis2 = ps.feature_library.fourier_library.FourierLibrary(n_frequencies=6)
basis = basis1 + basis2
n_features = 20

#%% Running the algorithm

coeff_list = []
mask_list = []
error_list = []

print(f"Running the cross-validated SSR algorithm for variable trajectory length.\n\
    dt = {dt}, with {nnum} different trajectory lengths ranging from {10**npow_lo} to\
    {10**npow_hi}, and {n_traj} trajectories per sample. \n\n\
    Data collected in {bin_n} bins. CV performed with {K} folds of data. \n\
    Results averaged over {n_sample} runs.")
time.sleep(0.5)

pbar = tqdm(ns, leave = False)

for n in pbar:
    errors = np.zeros(n_features)
    for i in range(n_sample):
        pbar.set_postfix_str(f'traj_len: {n}, sample: {i+1}/{n_sample}')
        # print(f"Computing: {n} points per trajectory -- run ({i})")
        
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
        
        # basis.fit(x_single)
        # X_single = basis.transform(x_single)
        
        # Computing the binned matrices
        x_binned, Y_binned, weights = bin_data(x_single, Y_single, bin_n,
                                           width_type = 'equal')
        basis.fit(x_binned)
        X_binned = basis.transform(x_binned)
        W = np.diag(weights)
        
        # Running the SSR algorithm
        
        errors += np.sqrt(CV_SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned), K = K))
        # coeffs, masks, errors = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned))
        
    error_list.append(errors/n_sample)
time.sleep(0.5)
print("Done!")
# coeff_list = np.array(coeff_list)
# mask_list = np.array(mask_list)
error_list = np.array(error_list)

#%% Visualising the CV scores

fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [20, 1]})

cmap = pl.cm.jet
norm = matplotlib.colors.LogNorm(vmin=10**npow_lo, vmax=10**npow_hi)
colors = pl.cm.jet(norm(ns).data)

ax[0].set_title("CV scores for different sample sizes")
ax[0].set_xlabel("Number of non-zero terms")
ax[0].set_ylabel(r"Mean CV score $\langle\delta\rangle$")
ax[0].set_xticks(np.arange(1,X_binned.shape[1]+1,1))
ax[0].set_xlim([1,X_binned.shape[1]])

for i in range(nnum):
    ax[0].semilogy(np.arange(1,X_binned.shape[1]+1,1), error_list[i], '.-',
                 color=colors[i], alpha = 0.9, markersize = 2)

matplotlib.colorbar.ColorbarBase(ax = ax[1], cmap = cmap, norm = norm, orientation = "vertical")
ax[1].invert_yaxis()
ax[1].set_ylabel("Trajectory length")

plt.savefig("example1_stat/cv_scores.pdf")
plt.show()