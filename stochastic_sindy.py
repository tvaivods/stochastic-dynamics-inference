import numpy as np
import pysindy as ps
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dataprep import *

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

def norm(X,Y,C):
    return np.linalg.norm(Y - np.matmul(X,C), ord = 2)

""" Stepwise Sparse Regressor """

def SSR_step(X,Y,mask = None):
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

def SSR(X, Y):
    basis_len = X.shape[1]
    dim = Y.shape[1]
    
    masks = [np.ones([basis_len, dim], dtype = bool)]
    mask = None
    coeffs = []
    errors = []
    
    for i in range(dim * basis_len):
        C, mask = SSR_step(X, Y, mask)
        masks.append(mask)
        coeffs.append(C)
        errors.append(norm(X, Y, C))
    
    masks = np.flip(np.array(masks[:-1], dtype = bool), axis = 0)
    coeffs = np.flip(coeffs, axis = 0)
    errors = np.flip(errors)
    return coeffs, masks, errors
            
def opt_term_n(errors):
    """
    Estimates the optimal term number from an error array.

    Parameters
    ----------
    errors : array
        Array containing error values at positions i corresponding to the
        non-zero term number opt_n=i+1.

    Returns
    -------
    opt_n : int
        Optimal number of terms opt_n.

    """
    error_ratios = errors[:-1]/errors[1:]
    opt_n = np.argmax(error_ratios[:-1]/error_ratios[1:])+2
    return opt_n

def survival_to_order(masks):
    """
    Converts a survival matrix (i.e. an array of masks) into an array of
    survival orders.

    Parameters
    ----------
    masks : ndarray, dtype = bool
        Array of masks representing the survival matrix.

    Returns
    -------
    order : array
        1D array with each entry representing the survival position of their
        respective basis function. Survival position 1 corresponds to the
        term that survives the longest.

    """
    N = masks.shape[0]
    temp_masks = np.copy(masks)
    order = np.zeros(N)
    for i in range(N):
        pos = np.where(temp_masks[i])[0][0]
        order[pos] = i+1
        temp_masks[:,pos] = False
    return order

def order_to_survival(order):
    """
    Converts a survival order into a survival matrix.

    Parameters
    ----------
    order : array
        1D array with each entry representing the survival position of their
        respective basis function. Survival position 1 corresponds to the
        term that survives the longest.

    Returns
    -------
    masks : ndarray, dtype = bool
        Array of masks representing the survival matrix.

    """
    N = len(order)
    masks = np.zeros([N,N], dtype = bool)
    for i in range(N):  # for each function
        survival_pos = order[i]
        masks[survival_pos-1:,i] = True
    return masks

def CV_scores(X,Y,K=5):
    """
    Computes the cross-validation score values for each step in the
    SSR algorithm.

    Parameters
    ----------
    X : ndarray
        Matrix X.
    Y : ndarray
        Matrix y.
    K : int, optional
        Cross-validation fold number. The default is 5.

    Returns
    -------
    delta2 : array
        Squares of the CV scores.

    """
    
    kf8 = KFold(n_splits = K, shuffle = True, random_state = np.random.randint(1000))
    idx = np.arange(0,Y.shape[0])
    N = Y.shape[1] * X.shape[1]
    delta2 = np.zeros(N)
    
    for train_idx, test_idx in kf8.split(idx):
        mask = None
        for i in range(N):
            C, mask = SSR_step(X[train_idx], Y[train_idx], mask)
            delta2[N-i-1] += np.linalg.norm(
                                Y[test_idx]-np.matmul(X[test_idx],C),ord = 2)**2
    delta2 /= K*Y.shape[0]
    return delta2
