#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 10:46:15 2021

@author: tomass
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False
})
#%% Importing the data

suffix = f"_3-7(9n20s)"
basis_name = 'standard'
Ds = np.linspace(0,1,9)

data_tables = []
for D in Ds:
    filename = "stat_analysis/errdata"+suffix+f"D{D:.3f}-"+basis_name+".csv"
    data_table = np.loadtxt(filename, delimiter = ',')
    data_tables.append(data_table)
data_tables = np.array(data_tables)

ns = np.unique(data_tables[0,:,0])
n_lengths = len(ns)
n_strands = int(data_tables.shape[1]/n_lengths)
error_lists = np.sqrt(data_tables[:,:,1:])
n_features = error_lists.shape[2]
n_Ds = len(Ds)

# error_lists.reshape()

#%% Errors for fixed trajectory length

n_idx = 8

mean_lists = []
std_lists = []
for i in range(n_lengths):
    mean_lists.append(error_lists[
        :,int(i*n_strands):int((i+1)*n_strands),:].mean(axis=1))
    std_lists.append(error_lists[
        :,int(i*n_strands):int((i+1)*n_strands),:].std(axis=1))
mean_lists = np.transpose(mean_lists, (1,0,2))
std_lists = np.transpose(std_lists, (1,0,2))

fig, ax = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [20, 1]}, dpi=200)

cmap = pl.cm.jet
norm = matplotlib.colors.Normalize(vmin=Ds.min(), vmax=Ds.max())
colors = cmap(norm(ns))

ax[0].set_title(f"CV scores for different diffusion coefficients for $n\sim${5*ns[n_idx]:.2e}")
ax[0].set_xlabel("Sparsity level $n$")
ax[0].set_ylabel(r"Mean CV score $\langle \delta \rangle$")
ax[0].set_xticks(np.arange(1,mean_lists.shape[2]+1,1))
ax[0].set_xlim([1,mean_lists.shape[2]])
ax[0].grid()


for i in range(n_Ds):
    ax[0].plot(np.arange(1,n_features+1,1), mean_lists[i,n_idx,:], c = cmap(Ds[i]/Ds.max()))
    # ax[0].plot(np.arange(1,n_features+1,1), error_lists)

ax[0].set_yscale("log")
matplotlib.colorbar.ColorbarBase(ax = ax[1], cmap = cmap, norm = norm,
                                 orientation = "vertical")
ax[1].invert_yaxis()
ax[1].set_ylabel("Diffusion coefficient $D$")

fig.tight_layout()

plt.show()
