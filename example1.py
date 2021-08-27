#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt of recreating the 1D example from L. Boninsegna and C. Clementi,
“Sparse learning of stochastic dynamical equations,”
J. Chem. Phys., vol. 148, p. 241723, 2018, doi: 10.1063/1.5018409.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataprep import *
from stochastic_sindy import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "pgf.texsystem": "pdflatex",
    'pgf.rcfonts': False
})
#%% Defining the force function
y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3
u = lambda x : x**4/2-4*x**3+9*x**2-3*x
# y = lambda x : 0*x
x = np.linspace(-1,5,100)

fig, (ax1, ax2) = plt.subplots(1,2,dpi=200, figsize = (10,4))

ax1.plot(x,y(x), 'k')
ax1.set_xlabel("$x$")
ax1.set_ylabel("$F(x)$")
ax1.set_ylim([-35,30])
ax1.axhline(0, alpha = 0.2, c = 'k')
ax2.plot(x,u(x), 'k')
ax2.set_xlabel("$x$")
ax2.set_ylabel("$U(x)$")
ax2.set_ylim([-1,16])
plt.show()

#%% Generating time series of multiple trajectories
import multiprocessing as mp
import time

x_multiple = []
dt = 5e-3
n = int(1e3)
n_traj = 5
diffusion = 1
sigma_y = 0
sigma = sigma_y*dt/np.sqrt(2)
# x0s = np.full(4,2)
x0s = np.linspace(-0.606,4.27,n_traj)
t = time.time()

def mp_ts(x0):
    np.random.seed()
    return time_series(y,x0,dt,n,diffusion)
    
pool = mp.Pool()
x_multiple = pool.map(mp_ts, x0s)
pool.close()
pool.join()

# for x0 in np.linspace(-1,5,n_traj):
#     x_multiple.append(time_series(y, x0, dt, n, diffusion))

x_multiple = np.array(x_multiple)

print(time.time() - t)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (6,4), dpi = 200)
plt.rcParams['axes.facecolor'] = 'w'

for i in range(n_traj):
    plt.plot(np.linspace(0,n*dt,n+1),x_multiple[i], linewidth = 0.3)
plt.xlabel("$t$")
plt.ylabel("$x$")

# x_multiple = add_noise(x_multiple, sigma)

# for i in range(n_traj):
#     plt.scatter(x_multiple[i],np.linspace(0,n*dt,n+1), c='r', s=4)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.show()

x_multiple = np.array(x_multiple).T

# x_multiple = add_noise(x_multiple, .05)

#%% Trajectory plot

import multiprocessing as mp
import time

fig, axs = plt.subplots(1,3, figsize = (6,4), dpi = 200, sharey=True)
plt.rcParams['axes.facecolor'] = 'w'

Ds = [0.2,1,5]

for di in range(3):
    diffusion = Ds[di]
    x_multiple = []
    dt = 5e-3
    n = int(1e3)
    n_traj = 5
    sigma_y = 0
    sigma = sigma_y*dt/np.sqrt(2)
    # x0s = np.full(4,2)
    x0s = np.linspace(-0.606,4.27,n_traj)
    t = time.time()
    
    def mp_ts(x0):
        np.random.seed()
        return time_series(y,x0,dt,n,diffusion)
        
    pool = mp.Pool()
    x_multiple = pool.map(mp_ts, x0s)
    pool.close()
    pool.join()
    
    # for x0 in np.linspace(-1,5,n_traj):
    #     x_multiple.append(time_series(y, x0, dt, n, diffusion))
    
    x_multiple = np.array(x_multiple)
    
    print(time.time() - t)
    
    
    
    for i in range(n_traj):
        axs[di].plot(x_multiple[i], np.linspace(0,n*dt,n+1),linewidth = 0.3)
    axs[di].set_xlabel("$x$")
    axs[di].set_xlim([-3,7])

# x_multiple = add_noise(x_multiple, sigma)

# for i in range(n_traj):
#     plt.scatter(x_multiple[i],np.linspace(0,n*dt,n+1), c='r', s=4)
# plt.xlabel("x")
# plt.ylabel("t")
# plt.show()
fig.tight_layout()
axs[0].set_ylabel("$t$")

plt.show()
x_multiple = np.array(x_multiple).T

# x_multiple = add_noise(x_multiple, .05)
#%% Computing the matrices X and Y
Y_multiple = ps.differentiation.FiniteDifference(order = 1)._differentiate(
    x_multiple, dt
)

# Stack the data into one column
Y_single = Y_multiple.reshape(-1, 1, order = 'F')
x_single = x_multiple.reshape(-1, 1, order = 'F')

basis = ps.feature_library.polynomial_library.PolynomialLibrary(degree=15)
basis.fit(x_single)
X_single = basis.transform(x_single)

n_pos = X_single.shape[1]*Y_single.shape[1]
n_features = X_single.shape[1]

#%% Bin the data and use it to infer the force
x_binned, Y_binned, weights = bin_data(x_single, Y_single, 90,
                                       width_type = 'equal')
basis.fit(x_binned)
X_binned = basis.transform(x_binned)
W = np.diag(weights)

plt.scatter(x_binned.squeeze(), Y_binned.squeeze(), s = 40*(weights/np.max(weights)), 
            marker = 'o', facecolor = 'b', alpha = 0.3)
plt.plot(x,y(x), 'k')


mask = np.zeros((n_features,1), dtype = 'bool')
mask[0:4] = True
# mask = np.ones([True, True, True, True, False, False,
#                  False, False, False, False, False]).reshape(-1,1)
C, mask = SSR_step(np.matmul(W,X_binned), np.matmul(W,Y_binned), mask)
print(C)

y_pred = lambda x : C[0] + C[1]*x + C[2]*x**2 + C[3]*x**3 + C[4]*x**4
plt.plot(x, y_pred(x), 'r--')
plt.show()

#%%
errors = CV_scores(np.matmul(W,X_binned), np.matmul(W,Y_binned), K=5)
plt.semilogy(np.arange(1, X_single.shape[1] * Y_single.shape[1] + 1, 1), errors, '-o')
plt.xticks(np.arange(1,X_single.shape[1]+1,1))
plt.xlabel("Total number of nonzero terms")
plt.ylabel("Cross-validation score $\delta$")
plt.show()

opt_n = opt_term_n(errors)
print(opt_n)

#%%
Cs, masks, errors = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned))
coeffs = np.abs(Cs.squeeze())
coeffs[~masks.squeeze()] = None
# for i in range(coeffs.shape[0]):
#     coeffs[i,:] -= np.nanmin(coeffs[i,:])*.99
#     coeffs[i,:] /= np.nanmax(coeffs[i,:])
coeffs = np.log10(coeffs)

###
idxs = np.arange(0.5,n_features+1.5,1)
pos = np.arange(0.5,n_pos+1.5,1)

X_mesh, Y_mesh = np.meshgrid(idxs, pos)

plt.rcParams['axes.facecolor'] = [.65]*3

fig = plt.figure(dpi = 200)
ax = fig.add_subplot(1,1,1)

cmap = pl.cm.Reds.copy()
cmap.set_bad(alpha = 0)

plt.rc('axes', axisbelow=False)

im = plt.pcolor(X_mesh, Y_mesh, coeffs[:,:], cmap = cmap,
                  shading = "flat", alpha = 0.95)
plt.gca().invert_yaxis()

# x-axis
def insert(source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

f_names = basis.get_feature_names(["x","y"])
f_names = np.ravel(f_names)
f_names = np.array(["$"+insert(s,"{",s.find("^")+1)+"}$" for s in f_names])
plt.xticks(np.arange(1,n_pos+1,1),
           f_names,
           rotation='vertical')
ax.set_xticks(np.arange(0.5,n_pos+1,1), minor=True)

# y-axis
y_labels = np.arange(1,n_pos+1,1).astype('str')
y_labels = np.array(["$"+s+"$" for s in y_labels])
plt.yticks(np.arange(1,n_pos+1,1), y_labels)
ax.set_yticks(np.arange(0.5,n_pos+1,1), minor=True)
plt.ylim([n_pos+0.5,0])
plt.xlim([0.5, n_pos+0.5])
for i in range(n_pos):
    if true_mask.ravel(order="F")[i]:
        pass
        # plt.axvline(i+0.5, zorder = 1, c = 'k')
        # plt.axvline(i+1.5, zorder = 1, c = 'k')
        # f_name = basis.get_feature_names(["x","y"])[np.mod(i, X.shape[1])]
        # plt.text(i+.75,-.4,"$"+f_name+"$")

# Optimal functions
# plt.axhline(opt_n-.5, c="k", zorder = 10)
# plt.axhline(opt_n+.5, c="k", zorder = 10)
opt_x, opt_y = np.meshgrid([0.5, n_pos+0.5], [opt_n-.5, opt_n+.5])
plt.pcolor(opt_x, opt_y, np.full((1,1),1), alpha = 0.5, shading = 'flat',
           cmap = 'binary', zorder = -1)

plt.xlabel("Function")
plt.ylabel("Survival position")
plt.grid(True, which = "minor", linewidth = 1, c = 'w')
ax.tick_params(length = 0, which = 'both')
ax.set_aspect("equal")

# plt.scatter(np.arange(1, n_pos+1,1)[true_mask.ravel(order = "F")],
#             [5]*true_mask.sum(), c = "w")
# plt.grid()
plt.show()