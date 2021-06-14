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
x = np.linspace(-1,5,100)
plt.plot(x,y(x))
plt.show()

#%% Generating time series of multiple trajectories
x_multiple = []
dt = 5e-4
n = 600
n_traj = 5
diffusion = .4

for x0 in np.linspace(-1,5,n_traj):
    x_multiple.append(time_series(y, x0, dt, n, diffusion))

for i in range(n_traj):
    plt.plot(x_multiple[i],np.linspace(0,n*dt,n+1))
plt.xlabel("x")
plt.ylabel("t")
plt.show()

x_multiple = np.array(x_multiple).T
#%% Computing the matrices X and Y
Y_multiple = ps.differentiation.FiniteDifference()._differentiate(
    x_multiple, dt
)

# Stack the data into one column
Y_single = Y_multiple.reshape(-1, 1, order = 'F')
x_single = x_multiple.reshape(-1, 1, order = 'F')

basis = ps.feature_library.polynomial_library.PolynomialLibrary(degree=4)
basis.fit(x_single)
X_single = basis.transform(x_single)

#%% Bin the data and use it to infer the force
x_binned, Y_binned, weights = bin_data(x_single, Y_single, 50, width_type = 'equal')
basis.fit(x_binned)
X_binned = basis.transform(x_binned)
W = np.diag(weights)

plt.scatter(x_binned.squeeze(), Y_binned.squeeze(), s = 40*(weights > 0), 
            marker = 'o', facecolor = 'b', alpha = 0.3)
plt.plot(x,y(x), 'k')

mask = np.array([True, True, True, True, False]).reshape(-1,1)
C, mask = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned), mask)
# C, mask = SSR(X_binned,Y_binned, mask)
print(C)

y_pred = lambda x : C[0] + C[1]*x + C[2]*x**2 + C[3]*x**3 + C[4]*x**4
plt.plot(x, y_pred(x), 'r--')
plt.show()

#%%
# mask = None
# for i in range(4):
#     C, mask = SSR(np.matmul(W,X_binned), np.matmul(W,Y_binned), mask)
#     print("C:\n", C)
#     print("Score: ", np.linalg.norm(Y_single - np.matmul(X_single, C), ord = 2)/Y_single.shape[0])

#%%
errors = CV_score(np.matmul(W,X_binned), np.matmul(W,Y_binned), K=10)
plt.plot(np.arange(0, X_single.shape[1] * Y_single.shape[1], 1), errors, '-o')
plt.ylim(0, np.max(errors)*1.1)
plt.xlabel("Total number of nonzero terms")
plt.ylabel("Cross-validation score $\delta$")
plt.show()
