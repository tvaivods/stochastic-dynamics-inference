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

#%% Loading the data

suffix = '_3-5(3n20s)D0.500-standard'
error_list = np.loadtxt("stat_analysis/errdata"+suffix+".csv", delimiter = ',')
n_lengths = len(np.unique(error_list[:,0]))
n_strands = error_list.shape[0]/n_lengths
n_features = error_list.shape[1]
ns = np.unique(error_list[:,0])

#%% Visualising the CV scores

mean_list = []
std_list = []
for i in range(n_lengths):
    mean_list.append(error_list[int(i*n_strands):int((i+1)*n_strands), 1:].mean(axis=0))
    std_list.append(error_list[int(i*n_strands):int((i+1)*n_strands), 1:].std(axis=0))
mean_list = np.array(mean_list)
std_list = np.array(std_list)

# Plotting the data
fig, ax = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [20, 1]}, dpi=200)

cmap = pl.cm.jet
norm = matplotlib.colors.LogNorm(vmin=np.min(error_list[:,0]),
                                 vmax=np.max(error_list[:,0]))
colors = cmap(norm(error_list[:,0]).data)

ax[0].set_title("CV scores for different sample sizes")
ax[0].set_xlabel("Number of non-zero terms")
ax[0].set_ylabel(r"CV score $\delta^2$")
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

#%% CV score v. Traj length for fixed n. of terms

import sklearn as sk

fig = plt.figure(figsize=(8, 4), dpi=200)

n_terms_list = [3,4,5]

reg = sk.linear_model.LinearRegression(fit_intercept = True)
reg.fit(np.log(ns.reshape(-1,1)), np.log(mean_list))
coeffs = reg.coef_
intercepts = reg.intercept_

for n_terms in n_terms_list:

    c = coeffs[n_terms - 1]
    a = intercepts[n_terms - 1]
    
    # for i in range(n_lengths):
    #     plt.scatter(ns[i], mean_list[i,3])
    plt.scatter(ns, mean_list[:,n_terms-1])
    plt.yscale("log")
    plt.xscale("log")
    
    plt.plot(ns, np.exp(a)*ns**c, '--', alpha = 0.5)

plt.legend([rf"${n_terms}$" for n_terms in n_terms_list],
           title = "# of non-zero terms")
plt.ylabel("$\delta^2$")
plt.xlabel("Trajectory length")

plt.show()

#%% Slope v. n. of nonzero terms

fig = plt.figure(figsize=(8, 4), dpi=200)

plt.scatter(np.arange(1,n_features,1), coeffs, c = 'k', s = 15)

plt.title("Change in the exponent of the fit of $\delta^2 \propto n^c$")
plt.ylabel("$Exponent c$")
plt.xlabel("Number of non-zero terms")
plt.xticks(np.arange(1,n_features,1))


