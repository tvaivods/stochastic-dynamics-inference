#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:34:55 2021

@author: tomass
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt of recreating the 1D example from L. Boninsegna and C. Clementi,
“Sparse learning of stochastic dynamical equations,”
J. Chem. Phys., vol. 148, p. 241723, 2018, doi: 10.1063/1.5018409.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataprep import *
from stochastic_sindy import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
#%% Defining the force function
def v(X):
    x = X[0]
    y = X[1]
    m = .8
    vx = y - m*x**3/2
    vy = m*y*(1-x**2)-x
    return np.array([vx,vy])

dt = 5e-3
n = int(1e4)
diffusion = 1.2
# sigma_y = 1
# sigma = sigma_y*dt/np.sqrt(2)
meas_to_stoch = 0
sigma = meas_to_stoch*np.sqrt(2*dt)
print(f"sqrt(2 dt)D, sigma = {np.sqrt(2*dt)}, {sigma}")
# sigma = 0.02
x0 = [1.25,1]
# x0 = [0,0]
# x0s = np.linspace(-0.606,4.27,n_traj)

xs = time_series(v, x0, dt, n, diffusion)
xs = add_noise(xs, sigma)
xs_det = time_series(v,x0,dt,1500)

fig, axs = plt.subplots(1,2,dpi = 200, gridspec_kw={'width_ratios': [1, 1]},
                        sharey = True, figsize = (6,8))
for ax in axs:
    ax.set_facecolor("w")


# Stramplot
X, Y = np.meshgrid(np.linspace(xs[:,0].min()-1, xs[:,0].max()+1,100),
                   np.linspace(xs[:,1].min()-1, xs[:,1].max()+1,100))
U, V = v([X,Y])[0], v([X,Y])[1]
axs[0].streamplot(X,Y,U,V, color = (U**2+V**2)**(1/2), cmap = 'jet',
               linewidth = 1, arrowsize = 0.75)
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$y$")
axs[0].axis([X.min(), X.max(), Y.min(), Y.max()])
axs[0].set_aspect('equal', adjustable='box')


axs[1].plot(xs[:,0],xs[:,1], c = 'green', linewidth = 0.15)
axs[1].plot(xs_det[:,0], xs_det[:,1], linestyle = "--", c = 'k', linewidth = 1)
axs[1].set_xlabel("$x$")
axs[1].axis([X.min(), X.max(), Y.min(), Y.max()])
axs[1].set_aspect('equal', adjustable='box')


fig.tight_layout()

plt.show()

#%% Computing the matrices X and Y
Y = ps.differentiation.FiniteDifference(order = 1)._differentiate(
    xs, dt
)

basis = ps.feature_library.polynomial_library.PolynomialLibrary(degree=3)
basis.fit(xs)
X = basis.transform(xs)

n_pos = X.shape[1]*Y.shape[1]
n_features = X.shape[1]

# Set the true mask
#['1', 'x', 'y', 'x^2', 'x y', 'y^2', 'x^3', 'x^2 y', 'x y^2', 'y^3']
mask_x = np.array([False, False, True, False, False, False, True, False, False,
                  False])
mask_y = np.array([False, True, True, False, False, False, False, True, False,
                  False])
true_mask = np.array([mask_x, mask_y]).T

#%% Determine optimal number of terms
errors = CV_scores(X, Y, K=100)
# errors = np.log(errors)

fig, ax = plt.subplots(1,1,figsize = (4,3), dpi = 200)
ax.set_facecolor("w")
plt.plot(np.arange(1, X.shape[1] * Y.shape[1] + 1, 1), errors, '-o',
             markersize=3)
plt.xticks(np.arange(1,n_pos+1,1))
plt.xlabel("Total number of nonzero terms")
plt.ylabel("Cross-validation score $\delta$")


opt_n = opt_term_n(errors)
plt.axvline(opt_n, c='r', linestyle = '-')
print("Ratio: ",opt_n)
opt_n = opt_term_n(errors, method = "laplacian")
plt.axvline(opt_n, c ='g', linestyle = '-.')
print("Laplacian: ", opt_n)
# opt_n = opt_term_n(errors, method = "edge")
# plt.axvline(opt_n, c ='b', linestyle = '--')
# print("Edge: ", opt_n)

plt.show()

#%%
Cs, masks, errors = SSR(X, Y)
coeffs = np.abs(Cs.squeeze())
coeffs[~masks.squeeze()] = None
# for i in range(coeffs.shape[0]):
#     coeffs[i,:] -= np.nanmin(coeffs[i,:])*.99
#     coeffs[i,:] /= np.nanmax(coeffs[i,:])
coeffs = np.log10(coeffs)
n_pos = X.shape[1]*Y.shape[1]

idxs_x = np.arange(0.5,X.shape[1]+1.5,1)
idxs_y = np.arange(X.shape[1]+0.5, n_pos+1.5,1)
pos = np.arange(0.5,n_pos+1.5,1)

X_mesh_x, Y_mesh = np.meshgrid(idxs_x, pos)
X_mesh_y, Y_mesh = np.meshgrid(idxs_y, pos)

plt.rcParams['axes.facecolor'] = [.65]*3

fig = plt.figure(figsize = (6,6), dpi = 200)
ax = fig.add_subplot(1,1,1)

x_cmap = pl.cm.Reds.copy()
x_cmap.set_bad(alpha = 0)
y_cmap = pl.cm.Blues.copy()
y_cmap.set_bad(alpha = 0)

plt.rc('axes', axisbelow=False)

im_x = plt.pcolor(X_mesh_x, Y_mesh, coeffs[:,:,0], cmap = x_cmap,
                  shading = "flat", alpha = 0.95)
im_y = plt.pcolor(X_mesh_y, Y_mesh, coeffs[:,:,1], cmap = y_cmap,
                  shading = 'flat', alpha = 0.95)
plt.gca().invert_yaxis()

# x-axis
f_names = basis.get_feature_names(["x","y"])
f_names = np.array(["$"+s+"$" for s in f_names])
plt.xticks(np.arange(1,n_pos+1,1),
           f_names[np.mod(np.arange(0,n_pos,1),X.shape[1], dtype = np.int32)],
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
plt.axhline(opt_n, c="w", zorder = 10)
# opt_x, opt_y = np.meshgrid([0.5, n_pos+0.5], [opt_n-.5, opt_n+.5])
# plt.pcolor(opt_x, opt_y, np.full((1,1),1), alpha = 0.5, shading = 'flat',
           # cmap = 'binary', zorder = -1)

# True functions
true_idxs = np.arange(n_pos)[true_mask.ravel(order = "F")]
for idx in true_idxs:
    plt.axvline(idx+1, c = 'w')

plt.xlabel("Function")
plt.ylabel("Survival position")
plt.grid(True, which = "minor", linewidth = 0.25, c = 'w')
ax.tick_params(length = 0, which = 'both')
ax.set_aspect("equal")

# plt.scatter(np.arange(1, n_pos+1,1)[true_mask.ravel(order = "F")],
#             [5]*true_mask.sum(), c = "w")
# plt.grid()
plt.show()

# plt.plot(np.arange(1,X.shape[1]*2+1,1), errors, '-o')


