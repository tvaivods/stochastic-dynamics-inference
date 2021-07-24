#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 10:46:15 2021

@author: tomass
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#%% Importing the data

suffix = f"_3-5(3n20s)"
basis_name = 'standard'
Ds = np.linspace(0,1,5)

data_tables = []
for D in Ds:
    filename = "stat_analysis/errdata"+suffix+f"D{D:.3f}-"+basis_name+".csv"
    data_table = np.loadtxt(filename, delimiter = ',')
    data_tables.append(data_table)
data_tables = np.array(data_tables)

ns = np.unique(data_tables[0,:,0])
n_lengths = len(ns)
n_strands = int(data_tables.shape[1]/n_lengths)
error_lists = data_tables[:,:,1:]
n_features = error_lists.shape[2]
n_Ds = len(Ds)

#%%

mean_lists = []
std_lists = []
for i in range(n_lengths):
    mean_lists.append(error_lists[
        :,int(i*n_strands):int((i+1)*n_strands),:].mean(axis=1))
    std_lists.append(error_lists[
        :,int(i*n_strands):int((i+1)*n_strands),:].std(axis=1))
mean_lists = np.transpose(mean_lists, (1,0,2))
std_lists = np.transpose(std_lists, (1,0,2))

cmap = pl.cm.jet

for i in range(n_Ds):
    plt.semilogy(np.arange(1,n_features+1,1), mean_lists[i,1,:], c = cmap(Ds[i]))
