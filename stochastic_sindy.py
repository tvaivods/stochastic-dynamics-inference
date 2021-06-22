import numpy as np
import pysindy as ps
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def infer_diffusion(x, dt):
    """ Estimates the diffusion coefficient in 1D. """
    a = (x[1:]-x[:-1])**2/2/dt
    return np.sqrt(np.mean(a))
         

def LSQ(X, y):
    """ Least-squares linear regression to find c = argmin_xi ||y - X xi||_2 """
    reg = sk.linear_model.LinearRegression(fit_intercept = False)
    reg.fit(X, y)
    c = reg.coef_.T
    return c

""" Stepwise Sparse Regressor """

def SSR(X,Y,mask = None):
    """
    One step of the Stepwise Sparse Regressor. Minimises ||Y-X C||_2 across specific entries
    of C, indicated by mask. Returns the optimal coefficients C and a new mask that blocks out
    the coefficient with the lowest absolute value.

    Parameters
    ----------
    Y : array of shape [N,d].
    X : array of shape [N,K].
    mask : boolean array of shape [K,d]
        Indicates which entries of the solution can be nonzero.
        If no mask is provided, considers all entries.

    Returns
    -------
    C : array of shape [K,d] of coefficients.
    new_mask : array of shape [K,d] of the next mask.
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

    if mask.any():
        for i in range(d):
            if (~mask[:,i]).all():
                pass
            else:
                # First solve least-squares for each column (1,...,d) for the non-masked coeffs.
                ci = LSQ(X[:, mask[:,i]], Y[:,i])
                C[mask[:,i],i] = ci
                if np.min(np.abs(ci)) < C_min:
                    C_min = np.min(np.abs(ci))
                    m = np.where(np.abs(C[:,i]) == C_min)[0][0] # row of the smallest entry
                    n = i   # col of the smallest entry
        new_mask[m,n] = False
    else:
        raise Exception("All coefficients are masked out")

    return C, new_mask

def survival_matrix(X,Y):
    N = Y.shape[1] * X.shape[1]
    mask_matrix = np.ones([X.shape[1], X.shape[1]])
    mask = None
    for i in range(N-1):
        _, mask = SSR(X, Y, mask)
        mask_matrix[N-2-i,:] = mask.squeeze()
    return mask_matrix
        
    

def CV_score(X,Y,K=8):
    """Computes the cross-validation score values for each step in the
    SSR algorithm.
    """
    kf8 = KFold(n_splits = K, shuffle = True, random_state = np.random.randint(1000))
    idx = np.arange(0,Y.shape[0])
    N = Y.shape[1] * X.shape[1]
    delta2 = np.zeros(N)
    
    for train_idx, test_idx in kf8.split(idx):
        mask = None
        for i in range(N):
            C, mask = SSR(X[train_idx], Y[train_idx], mask)
            delta2[N-i-1] += np.linalg.norm(
                                Y[test_idx]-np.matmul(X[test_idx],C),ord = 2)**2
    delta2 /= K
    return np.sqrt(delta2)
