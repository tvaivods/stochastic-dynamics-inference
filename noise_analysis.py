#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:34:54 2021

@author: tomass
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from dataprep import *
from stochastic_sindy import *
from bases import *
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.patches as patches

#%%
ledger = pd.read_csv("stat_analysis/noise_ledger.csv",
                     skipinitialspace=True)

#%% Update ledger with fitted parameters

for i in range(len(ledger)):
    if(ledger["trans_pos"].isna()[i]):
        D, s_lo, s_hi, n_Ns, n_strands = ledger.loc[i,["diffusion","s_lo","s_hi","s_num","sample_n"]]
        suffix = f"_6-6(1n{int(n_strands)}s)D{D:.3f}"
        basis_name = "standard"
        n_Ns = int(n_Ns)
        Ns = np.linspace(s_lo,s_hi,n_Ns)
        
        error_tables = []
        mask_tables = []
        order_tables = []
        
        for N in Ns:
            error_table = np.loadtxt("stat_analysis/errdata"+suffix+f"N{N:.3f}-"
                                +basis_name+".csv", delimiter = ',')
            error_tables.append(error_table)
            mask_table = np.loadtxt("stat_analysis/survmasks"+suffix+f"N{N:.3f}-"
                                +basis_name+".csv", delimiter = ',')
            mask_tables.append(mask_table)
            order_table = np.loadtxt("stat_analysis/survorders"+suffix+f"N{N:.3f}-"
                                +basis_name+".csv", delimiter = ',')
            order_tables.append(order_table)
        
        error_tables = np.array(error_tables)[:,:,1:]
        mask_tables = np.array(mask_tables)[:,:,1:].astype("bool")
        order_tables = np.array(order_tables)[:,:,1:].astype("int")
        ns = np.unique(mask_table[:,0])
        
        n_lengths = len(np.unique(mask_table[:,0]))
        n_strands = int(mask_table.shape[0]/n_lengths)
        n_features = mask_table.shape[1]-1
        
        true_mask = np.zeros(n_features, dtype = bool)
        true_mask[0:4] = True
        
        accuracy_lists = []
        for i_N in range(n_Ns):
            accuracy_list = []
            n_accurate = np.count_nonzero((true_mask == mask_tables[i_N]).all(axis = 1))
            accuracy_list.append(n_accurate)
            accuracy_lists.append(accuracy_list)
        accuracy_lists = np.array(accuracy_lists)/n_strands
        
        p = accuracy_lists.ravel()
        
        def sigmoid(x, a, b, c):
            return a/(1+np.exp(-b*(x-c)))
        
        popt, pcov = curve_fit(sigmoid, Ns, p)
        
        ledger.at[i,"trans_pos"] = popt[2]
        ledger.at[i,"trans_rate"] = popt[1]
        ledger.at[i, "p_init"] = popt[0]

ledger.to_csv("stat_analysis/noise_ledger.csv", index=False)

#%% Transition values

fig = plt.figure(figsize = (6,3), dpi = 200)

plt.plot(ledger["diffusion"], ledger["trans_pos"], 'k.')
plt.ylim([0, ledger["trans_pos"].max()+0.05])
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Critical noise coefficient, $\hat{\sigma}$")
plt.title("Transition values for function choice accuracy")
plt.grid()

#%% Transition rates

fig = plt.figure(figsize = (6,3), dpi = 200)

plt.semilogy(ledger["diffusion"], abs(ledger["trans_rate"]), 'k.')
# plt.ylim([0, abs(ledger["trans_rate"]).max()])
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Transition rate")
plt.title("Transition rates for function choice accuracy")
plt.grid()

#%%
D = 4
idx = ledger.index[ledger["diffusion"]==D][0]
basis_name = "standard"

s_lo, s_hi, n_Ns, n_strands = ledger.loc[idx,["s_lo","s_hi","s_num","sample_n"]]
n_noises = ledger.at[idx, "s_num"]
suffix = f"_6-6(1n{int(n_strands)}s)D{D:.3f}"
n_Ns  = int(n_Ns)
Ns = np.linspace(s_lo,s_hi,n_Ns)

error_tables = []
mask_tables = []
order_tables = []

for N in Ns:
    error_table = np.loadtxt("stat_analysis/errdata"+suffix+f"N{N:.3f}-"
                        +basis_name+".csv", delimiter = ',')
    error_tables.append(error_table)
    mask_table = np.loadtxt("stat_analysis/survmasks"+suffix+f"N{N:.3f}-"
                        +basis_name+".csv", delimiter = ',')
    mask_tables.append(mask_table)
    order_table = np.loadtxt("stat_analysis/survorders"+suffix+f"N{N:.3f}-"
                        +basis_name+".csv", delimiter = ',')
    order_tables.append(order_table)

error_tables = np.array(error_tables)[:,:,1:]
mask_tables = np.array(mask_tables)[:,:,1:].astype("bool")
order_tables = np.array(order_tables)[:,:,1:].astype("int")
ns = np.unique(mask_table[:,0])

n_lengths = len(np.unique(mask_table[:,0]))
n_strands = int(mask_table.shape[0]/n_lengths)
n_features = mask_table.shape[1]-1

true_mask = np.zeros(n_features, dtype = bool)
true_mask[0:4] = True

accuracy_lists = []
for i_N in range(n_Ns):
    accuracy_list = []
    n_accurate = np.count_nonzero((true_mask == mask_tables[i_N]).all(axis = 1))
    accuracy_list.append(n_accurate)
    accuracy_lists.append(accuracy_list)
accuracy_lists = np.array(accuracy_lists)/n_strands

# Fit sigmoid
p = accuracy_lists.ravel()

def sigmoid(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))

popt, pcov = curve_fit(sigmoid, Ns, p)

acc_var = np.sqrt(p*(1-p)/n_strands)

fig = plt.figure(figsize = (8,4), dpi = 200)

x = np.linspace(Ns.min(),Ns.max(),100)
plt.plot(x, sigmoid(x, *popt), c='k')
plt.errorbar(Ns, p, acc_var, fmt = '.',capsize=3, c='k')
for i in range(n_Ns):
    plt.plot(Ns[i], accuracy_lists[i].reshape(1,-1), c='k',
             marker = '.', markersize=10)
plt.ylim([-0.05, 1.05])
plt.text(Ns[0], 0.1, f"Transition value: {popt[2]:.3f}\nTransition rate:\
{popt[1]:.3f}")
plt.ylabel("Fraction of accurate optimal predictions")
plt.xlabel("Measurement noise coefficient $\sigma$")
plt.title(f"Sigmoid fit of accuracy frequency, D={D}")
plt.grid()

plt.axvline(popt[2], c = 'crimson', linestyle = '-')

#%% Sigmoid comparison

fig = plt.figure(figsize = (8,4), dpi = 200)

cmap = pl.cm.jet

x = np.linspace(0, ledger["s_hi"].max(), 200)
for i in range(len(ledger)):
    a = ledger.at[i,"p_init"]
    b = ledger.at[i,"trans_rate"]
    c = ledger.at[i,"trans_pos"]
    plt.plot(x, sigmoid(x, a,b,c), c=cmap(ledger.at[i, "diffusion"]/ledger["diffusion"].max()))
plt.legend([f"D={D}" for D in ledger["diffusion"]])
