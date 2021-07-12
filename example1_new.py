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

#%% Defining the force function
y = lambda x : 3 - 18*x  + 12*x**2 - 2*x**3
# y = lambda x : 0*x
x = np.linspace(-1,5,100)
plt.plot(x,y(x))
plt.show()

#%% Generating time series of multiple trajectories
import multiprocessing as mp
import time

x_multiple = []
dt = 5e-3
n = int(1e4)
n_traj = 5
diffusion = 1
x0s = np.full(n_traj, 2)
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

print(time.time() - t)

for i in range(n_traj):
    plt.plot(x_multiple[i],np.linspace(0,n*dt,n+1))
plt.xlabel("x")
plt.ylabel("t")
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

#%%

model = SSRSindy(x_single, Y_single)
basis = ps.feature_library.polynomial_library.PolynomialLibrary(degree=10)
model.fit_basis(basis)
model.bin_data(90)
model.evaluate()
coeffs = np.copy(np.abs(model.coeffs.squeeze()))
coeffs[~model.masks.squeeze()] = None
for i in range(coeffs.shape[0]):
    # coeffs[i,:] -= np.nanmin(coeffs[i,:])*.99
    coeffs[i,:] /= np.nanmax(coeffs[i,:])
coeffs = np.log10(coeffs)
plt.imshow(coeffs, cmap = 'viridis')
plt.colorbar()
plt.show()

plt.plot(np.arange(1,12,1), model.errors)
#%% Bin the data and use it to infer the force
x_binned, Y_binned, weights = bin_data(x_single, Y_single, 90,
                                       width_type = 'equal')
basis.fit(x_binned)
X_binned = basis.transform(x_binned)
W = np.diag(weights)

plt.scatter(x_binned.squeeze(), Y_binned.squeeze(), s = 40*(weights/np.max(weights)), 
            marker = 'o', facecolor = 'b', alpha = 0.3)
plt.plot(x,y(x), 'k')

mask = np.array([True, True, True, True, False]).reshape(-1,1)
C, mask = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned), mask)
# plt.scatter(x_single, Y_single, s = 0.2)
print(C)

y_pred = lambda x : C[0] + C[1]*x + C[2]*x**2 + C[3]*x**3 + C[4]*x**4
plt.plot(x, y_pred(x), 'r--')
plt.show()

#%%
errors = CV_score(np.matmul(W,X_binned), np.matmul(W,Y_binned), K=20)
plt.plot(np.arange(1, X_single.shape[1] * Y_single.shape[1] + 1, 1), errors, '-o')
plt.yscale("log")
# plt.ylim(0, np.max(errors)*1.1)
plt.xticks(np.arange(1,X_single.shape[1]+1,1))
plt.xlabel("Total number of nonzero terms")
plt.ylabel("Cross-validation score $\delta$")
plt.show()


