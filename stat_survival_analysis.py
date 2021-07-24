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
from bases import *

#%% Loading the data

suffix = f"_3-7(9n20s)"

mask_table = np.loadtxt("stat_analysis/survmasks"+suffix+".csv", delimiter = ',')
# mask_table = np.reshape(mask_table, (1,-1))
order_table = np.loadtxt("stat_analysis/survorders"+suffix+".csv", delimiter = ',')
# order_table = np.reshape(order_table, (1,-1))
n_lengths = len(np.unique(mask_table[:,0]))
n_strands = int(mask_table.shape[0]/n_lengths)
n_features = mask_table.shape[1]-1
ns = np.unique(mask_table[:,0])
mask_lists = mask_table[:,1:].reshape(n_lengths, n_strands, -1)
mask_lists = mask_lists.astype("bool")
order_lists = order_table[:,1:].reshape(n_lengths, n_strands, -1)

# True mask
true_mask = np.zeros(n_features, dtype = bool)
true_mask[0:4] = True

#%% Survival matrix plot
ns_idx = -1
opt_n = np.count_nonzero(true_mask)

order_list = order_lists[ns_idx]    # for the longest trajectories

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
plt.title(f"Mean survival pattern for trajectory of length {int(ns[ns_idx]):.1e}")

plt.errorbar(np.arange(1,n_features+1,1), order_mean, order_std, fmt = 'r.',
             markersize = 8, capsize = 4)

plt.ylim([n_features+0.5,0.5])

plt.savefig(f"stat_analysis/mean_survival.png", bbox_inches='tight')

#%%

ns_idx = -1

order_list = order_lists[ns_idx]

masks = []
for i in range(order_list.shape[0]):
    masks.append(order_to_survival(order_list[i]))
masks = np.array(masks)

surv_hist = masks.sum(axis = 0)
plt.imshow(surv_hist, cmap = 'Blues')

surv_hist = np.zeros((n_features,n_features))

for order in order_list:
    surv_hist[order.astype('int')-1, np.arange(0,n_features,1)] += 1

plt.imshow(surv_hist, cmap = 'Blues')
plt.xticks(np.arange(0,19,1))
plt.yticks(np.arange(0,19,1))
plt.grid(alpha = 0.3)


#%% Accurate function frequency

accuracy_list = []
for i in range(n_lengths):
    n_accurate = np.count_nonzero((true_mask == mask_lists[i]).all(axis = 1))
    accuracy_list.append(n_accurate)
accuracy_list = np.array(accuracy_list)

fig = plt.figure(figsize = (8,4), dpi = 200)

plt.scatter(ns, accuracy_list/n_strands)
plt.xscale("log")
plt.ylim([0, 1])
plt.ylabel(f"Fraction of accurate optimal predictions")
plt.xlabel("Trajectory length")
plt.title("Accuracy of function choice in optimal solution")
plt.grid()

# plt.savefig(f"stat_analysis/true_fraction.png", bbox_inches='tight')

#%% Hamming distances

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(n_lengths):
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

plt.savefig("stat_analysis/hamming_term_n.png", bbox_inches='tight')

#%% Jaccard distances

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    jac_coeff = np.count_nonzero(mask_lists[i] & true_mask, axis = 1)/\
        np.count_nonzero(mask_lists[i] | true_mask, axis = 1)
    avg_jac_dist = np.mean(1-jac_coeff)
    std_jac_dist = np.std(1-jac_coeff)
    plt.errorbar(ns[i], avg_jac_dist, std_jac_dist, fmt = 'k.', markersize = 8, capsize = 4)
plt.xscale("log")
plt.ylabel(f"Mean Jaccard distance (avg. over {n_strands} samples)")
plt.xlabel("Trajectory length")
plt.title("Jaccard distance between true and optimal function patterns")
plt.grid()
plt.ylim([0,1])

plt.savefig("stat_analysis/jaccard_term_n.png", bbox_inches='tight')

#%% Number of true and false functions in optimal solution

fig = plt.figure(figsize = (7,4), dpi = 200)

for i in range(len(ns)):
    n_true = np.count_nonzero(mask_lists[i,:,0:4], axis = 1)
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

plt.savefig("stat_analysis/true_term_n.png", bbox_inches='tight')

#%%

fig, ax = plt.subplots(1,2,figsize = (6,5),
                       gridspec_kw={'width_ratios': [20, 1]}, dpi=200)

cmap = pl.cm.jet
norm = matplotlib.colors.LogNorm(vmin=ns[0],
                                 vmax=ns[-1])
colors = cmap(norm(ns))

for i in range(len(ns)):
    n_match = np.count_nonzero(mask_lists[i] & true_mask, axis = 1)
    n_miss = np.count_nonzero(mask_lists[i] & ~true_mask, axis = 1)
    
    ax[0].errorbar(np.mean(n_match), np.mean(n_miss), np.std(n_match), np.std(n_miss),
                 alpha = 0.8, markersize = 8, capsize = 4, color = colors[i])
ax[0].grid()
ax[0].scatter(4,0, color = 'k', s = 40)
ax[0].set_ylabel("N. of incorrect functions")
ax[0].set_xlabel("N. of correct functions")
ax[0].set_title("Optimal solution function number")


matplotlib.colorbar.ColorbarBase(ax = ax[1], cmap = cmap, norm = norm, orientation = "vertical")
ax[1].invert_yaxis()
ax[1].set_ylabel("Trajectory length")

plt.savefig("stat_analysis/corr_v_incorr.png", bbox_inches='tight')