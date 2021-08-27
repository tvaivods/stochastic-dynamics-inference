#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:00:33 2021

@author: tomass
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from dataprep import *
from stochastic_sindy import *
from bases import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False
})
#%%
suffix = f"_3-7(9n20s)"
basis_name = "standard"
# Ds = np.linspace(0,2,9)
Ds = np.concatenate([np.linspace(0,2,9), [3,4,5]])
# np.array([0,1,5,10,15])
n_Ds = len(Ds)

mask_tables = []
order_tables = []

for D in Ds:
    mask_table = np.loadtxt("stat_analysis/survmasks"+suffix+f"D{D:.3f}-"
                            +basis_name+".csv", delimiter = ',')
    mask_tables.append(mask_table)
    order_table = np.loadtxt("stat_analysis/survorders"+suffix+f"D{D:.3f}-"
                            +basis_name+".csv", delimiter = ',')
    order_tables.append(order_table)

mask_tables = np.array(mask_tables)[:,:,1:].astype("bool")
order_tables = np.array(order_tables)[:,:,1:].astype("int")
ns = np.unique(mask_table[:,0])

n_lengths = len(np.unique(mask_table[:,0]))
n_strands = int(mask_table.shape[0]/n_lengths)
n_features = mask_table.shape[1]-1

masks = mask_tables.reshape(n_Ds, n_lengths, n_strands, n_features)
orders = order_tables.reshape(n_Ds, n_lengths, n_strands, n_features)

# True mask
true_mask = np.zeros(n_features, dtype = bool)
true_mask[0:4] = True

#%% Survival masks

ns_idxs = [0,5,-1]

fig, axs = plt.subplots(len(ns_idxs),n_Ds,figsize = (22,12),dpi = 200,
                        sharex = "all")
extent = [.5,n_features+.5,n_features+.5,.5]
cmaps = [pl.cm.Blues.copy(),
         pl.cm.GnBu.copy(),
         pl.cm.Greens.copy(),
         pl.cm.Oranges.copy(),
         pl.cm.Reds.copy()]

for j in range(n_Ds):
    for i in range(len(ns_idxs)):
        surv_hist = np.zeros((n_features,n_features))

        for order in orders[j,i]:
            surv_hist[order-1, np.arange(0,n_features,1)] += 1
        
        surv_hist_norm = surv_hist/np.max(surv_hist, axis = 0)
        surv_hist_norm[surv_hist == 0] = np.nan
        
        cmap = cmaps[j]
        cmap.set_bad(alpha = 0)
        
        axs[i,j].imshow(surv_hist_norm, cmap = cmap, extent = extent, vmin=0,
                        vmax = 1)
        axs[i,j].set_xticks(np.arange(1,n_features+1,1))
        axs[i,j].set_yticks(np.arange(1,n_features+1,1))
        axs[i,j].grid(alpha = 0.4)
        # axs[i,j].set_xlabel("Basis function index")
        # axs[i,j].set_ylabel("Survival order")
        # axs[i,j].set_title(f"{i,j}"

fig.tight_layout()

for ax, D in zip(axs[0], Ds):
    ax.set_title(f"$D$ = {D}")

for ax, n in zip(axs[:,0], ns[ns_idxs]):
    ax.set_ylabel(f"$n\sim$ {n:.2e}", size='large')

#%% Survival mask -- individual

ns_idx = -1
Ds_idx = -1

fig = plt.figure(figsize = (6,6),dpi = 200)
extent = [.5,n_features+.5,n_features+.5,.5]
cmap = pl.cm.Blues.copy()

surv_hist = np.zeros((n_features,n_features))

for order in orders[Ds_idx,ns_idx]:
    surv_hist[order-1, np.arange(0,n_features,1)] += 1

surv_hist_norm = surv_hist/np.max(surv_hist, axis = 0)
surv_hist_norm[surv_hist == 0] = np.nan

cmap.set_bad(alpha = 0)

plt.imshow(surv_hist_norm, cmap = cmap, extent = extent, vmin=0,
                vmax = 1)
plt.xticks(np.arange(1,n_features+1,1))
plt.yticks(np.arange(1,n_features+1,1))
plt.grid(alpha = 0.4)
plt.xlabel("Basis function index")
plt.ylabel("Survival order")
plt.title(f"Survival statistics for $n=${ns[ns_idx]} and $D=${Ds[Ds_idx]}")

fig.tight_layout()

#%% Accurate function frequency

accuracy_lists = []
for i_D in range(n_Ds):
    accuracy_list = []
    for i_n in range(n_lengths):
        n_accurate = np.count_nonzero((true_mask == masks[i_D, i_n]).all(axis = 1))
        accuracy_list.append(n_accurate)
    accuracy_lists.append(accuracy_list)
accuracy_lists = np.array(accuracy_lists)/n_strands

fig = plt.figure(figsize = (8,4), dpi = 200)

cmap = pl.cm.jet

for i in range(n_Ds):
    plt.plot(ns, accuracy_lists[i], c=cmap(Ds[i]/Ds.max()), marker = '.')
plt.xscale("log")
plt.ylim([-0.05, 1])
plt.ylabel(f"Fraction of accurate optimal predictions")
plt.xlabel("Trajectory length")
plt.title("Accuracy of function choice in optimal solution")
plt.grid()
plt.legend([f"$D$ = {D:.3f}" for D in Ds], loc = 'upper left')
    
#%% 