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
import glob

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

#%%
# ledger = pd.read_csv("stat_analysis/noise_ledger.csv",
#                      skipinitialspace=True)
ledger = pd.read_csv("stat_analysis/noise_ledger.csv", skipinitialspace=True)

#%% Update ledger with fitted parameters

for i in range(len(ledger)):
    D, n_strands = ledger.loc[i,["diffusion", "n_strands"]]
    suffix = f"_6-6(1n{int(n_strands)}s)D{D:.3f}"
    basis_name = "standard"
    
    titles = np.array(
        glob.glob(f"stat_analysis/errdata*{int(n_strands)}s*D{D:.3f}*.csv")
    )
    pos_of_D = np.array([s.find("N") for s in titles])
    Ns = [float(titles[i][pos_of_D[i]+1:pos_of_D[i]+6]) for i in range(len(pos_of_D))]
    Ns = np.array(Ns)
    n_Ns = len(Ns)
    
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
        return a/(1+np.exp(b*(x-c)))
    
    popt, pcov = curve_fit(sigmoid, Ns, p, p0 = [0,40,0.3])
    
    ledger["sigmas"] = ledger["sigmas"].astype("object")
    ledger.at[i, "sigmas"] = Ns
    ledger.at[i,"trans_pos"] = popt[2]
    ledger.at[i,"trans_rate"] = popt[1]
    ledger.at[i, "p_init"] = popt[0]
    ledger.at[i, "fit_error"] = np.sqrt(np.mean((sigmoid(Ns, *popt) - p)**2))

ledger = ledger.sort_values("diffusion")
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

plt.plot(ledger["diffusion"], abs(ledger["trans_rate"]), 'k.')
# plt.ylim([0, abs(ledger["trans_rate"]).max()])
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Transition rate, $r$")
plt.title("Transition rates for function choice accuracy")
plt.grid()


#%% Transition widths

fig = plt.figure(figsize = (6,3), dpi = 200)

widths = 2/ledger["trans_rate"]*np.log(0.99/(-0.99+1))

plt.plot(ledger["diffusion"], widths, 'k.')
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Transition width, $w$")
plt.title("Transition widths for function choice accuracy")
plt.grid()

#%% Unhindered probabilities

fig = plt.figure(figsize = (6,3), dpi = 200)

plt.plot(ledger["diffusion"], ledger["p_init"], 'k.')
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Unhindered probability $p_0$")
plt.title("Unhindered probabilities")
plt.grid()

#%%
Ds = [1,2,7,14]

fig = plt.figure(figsize = (8,4), dpi = 200)

cmap = pl.cm.jet

for D in Ds:
    idx = ledger.index[ledger["diffusion"]==D][0]
    basis_name = "standard"
    
    Ns, n_strands = ledger.loc[idx,["sigmas", "n_strands"]]
    n_Ns = len(Ns)
    suffix = f"_6-6(1n{int(n_strands)}s)D{D:.3f}"
    # n_Ns  = int(n_Ns)
    # Ns = np.linspace(s_lo,s_hi,n_Ns)
    
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
        return a/(1+np.exp(b*(x-c)))
    
    popt, pcov = curve_fit(sigmoid, Ns, p, p0 = [0,40,0.3])
    
    acc_var = np.sqrt(p*(1-p)/n_strands)

    x = np.linspace(0,Ns.max(),100)
    plt.plot(x, sigmoid(x, *popt), c=cmap(D/14), alpha = 0.75)
    plt.errorbar(Ns, p, acc_var, fmt = '.',capsize=3, alpha = 0.4, c=cmap(D/14))
    for i in range(n_Ns):
        plt.plot(Ns[i], accuracy_lists[i].reshape(1,-1),
                 marker = '.', markersize=10, c=cmap(D/14), alpha = 0.4)

# x = np.linspace(0, ledger.at[len(ledger)-1,"sigmas"].max()+0.05, 300)
# for i in range(len(ledger)):
#     a = ledger.at[i,"p_init"]
#     b = ledger.at[i,"trans_rate"]
#     c = ledger.at[i,"trans_pos"]
#     plt.plot(x, sigmoid(x, a,b,c), alpha = 0.75,
#              c=cmap(ledger.at[i, "diffusion"]/ledger["diffusion"].max()))

plt.ylim([-0.05, 1.05])

plt.ylabel("Fraction of accurate optimal predictions")
plt.xlabel("Measurement noise coefficient $\sigma$")
plt.title(f"Logistic fit of accuracy frequency, D={Ds}")

plt.grid()

#%% Sigmoid comparison

fig = plt.figure(figsize = (8,4), dpi = 200)

cmap = pl.cm.jet

x = np.linspace(0, ledger.at[len(ledger)-1,"sigmas"].max()+0.05, 300)
for i in range(len(ledger)):
    a = ledger.at[i,"p_init"]
    b = ledger.at[i,"trans_rate"]
    c = ledger.at[i,"trans_pos"]
    plt.plot(x, sigmoid(x, a,b,c), c=cmap(ledger.at[i, "diffusion"]/ledger["diffusion"].max()))
plt.legend([f"D={D}" for D in ledger["diffusion"]])


#%% Logistic fit errors

fig = plt.figure(figsize = (6,3), dpi = 200)

plt.plot(ledger["diffusion"], ledger["fit_error"], 'k.')
plt.xlim([0,ledger["diffusion"].max()+1])
plt.xlabel("Diffusion coefficient, $D$")
plt.ylabel(r"Logistic fit error")
plt.title("Logistic fit error v. $D$")
plt.grid()

#%% Fit the transition values

def f(x, a, b, c, d):
    return a*np.power(d*(x-b),c)

popt, pcov = curve_fit(f, ledger["diffusion"], ledger["trans_pos"],
                       [0.5,0.5,0.5,0.5])

plt.plot(ledger["diffusion"], ledger["trans_pos"], 'k.')
x = np.linspace(popt[0], ledger["diffusion"].max()+2, 100)
plt.plot(x, f(x,*popt))
plt.xlim([0,ledger["diffusion"].max()+2])
plt.grid()

#%% Transition subplots

fig, axs = plt.subplots(3,1,figsize = (6,10), dpi = 200)

axs[0].plot(ledger["diffusion"], ledger["trans_pos"], 'k.')
axs[0].set_ylim([0, ledger["trans_pos"].max()+0.05])
axs[0].set_xlim([0,ledger["diffusion"].max()+1])
axs[0].set_xlabel("Diffusion coefficient, $D$")
axs[0].set_ylabel(r"Critical noise coefficient, $\hat{\sigma}$")
axs[0].set_title("Transition values for function choice accuracy")
axs[0].grid()

axs[1].semilogy(ledger["diffusion"], abs(ledger["trans_rate"]), 'k.')
axs[1].set_xlim([0,ledger["diffusion"].max()+1])
axs[1].set_xlabel("Diffusion coefficient, $D$")
axs[1].set_ylabel(r"Transition rate, $r$")
axs[1].set_title("Transition rates for function choice accuracy")
axs[1].grid()

widths = 2/ledger["trans_rate"]*np.log(0.99/(-0.99+1))

axs[2].plot(ledger["diffusion"], widths, 'k.')
axs[2].set_xlim([0,ledger["diffusion"].max()+1])
axs[2].set_xlabel("Diffusion coefficient, $D$")
axs[2].set_ylabel(r"Transition width, $w$")
axs[2].set_title("Transition widths for function choice accuracy")
axs[2].grid()

fig.tight_layout()
