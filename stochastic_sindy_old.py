import numpy as np
import pysindy as ps
import sklearn as sk
from dynsys import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

""" Generating the time series data """
def lorenzsys(X, sigma=10, rho=28, beta=8/3):
    x, y, z = X
    v = np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])
    return v

dt = 0.01
T = 30
X0 = np.array([5,5,5])

dsys = DynamicalSystem(lorenzsys)
time_series = dsys.timeseries(X0, dt, int(T/dt), D=0.4)

scatter3(time_series)

""" Computing the velocities Y """
Y = ps.differentiation.FiniteDifference()._differentiate(time_series,dt)

""" Computing the matrix X """
basis = ps.feature_library.polynomial_library.PolynomialLibrary(degree=2)
basis.fit(time_series)
X = basis.transform(time_series)

def LSQ(X, y):
    """ Least-squares linear regression to find c = argmin_xi ||y - X xi||_2 """
    reg = sk.linear_model.LinearRegression()
    reg.fit(X, y)
    c = reg.coef_.T
    return c

""" Stepwise Sparse Regressor """

def SSR(X,Y,mask = None):
    """
    One step of the Stepwise Sparse Regressor. Minimises ||Y-X C||_2 across specific entries
    of C, indicated by mask. Returns the optimal coefficients C and a new mask that blocks out
    the coefficient with the lowest absolute value.
    Input:
        Y: array of shape [N,d].
        X: array of shape [N,K].
        mask: boolean array of shape [K,d] indicating which entries of the solution can be nonzero.
            If no mask is provided, considers all entries.
    Returns:
        C: array of shape [K,d] of coefficients.
        new_mask: array of shape [K,d] of the next mask.
    """
    d = Y.shape[1]
    N = Y.shape[0]
    K = X.shape[1]

    if type(mask) == type(None):
        mask = np.ones([K, d], dtype = bool)
    else:
        mask = np.array(mask, dtype = bool)
    new_mask = mask

    C = np.zeros([K, d])
    C_min = np.infty

    for i in range(d):
        if (~mask[:,i]).all():
            pass
        else:
            # First solve least-squares for each column (1,...,d) for the non-masked coeffs.
            ci = LSQ(X[:, mask[:,i]], Y[:,i])
            C[mask[:,i],i] = ci
            if np.min(np.abs(ci)) < C_min:
                C_min = np.min(np.abs(ci))
                m = np.argmin(np.abs(np.squeeze(C[:,i])) + 100*(~mask[:,i]))
                n = i
    new_mask[m,n] = False

    return C, new_mask

def CV_score(X,Y,K=8):
    kf8 = KFold(n_splits = K, shuffle = True, random_state = 42)
    idx = np.arange(0,Y.shape[0])

    N = X.shape[1] * Y.shape[1]
    delta2 = np.zeros(N)
    for train_idx, test_idx in kf8.split(idx):
        mask = None
        for i in range(N):
            C, mask = SSR(X[train_idx], Y[train_idx], mask)
            delta2[i] += np.linalg.norm(Y[test_idx]-np.matmul(X[test_idx],C), ord = 2)**2
    delta2 /= K
    return delta2


errors = CV_score(X,Y,10)
print(np.max(errors))
plt.plot(np.arange(X.shape[1] * Y.shape[1], 0, -1), errors, '-o')
plt.show()
