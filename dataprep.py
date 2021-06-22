import numpy as np
import matplotlib.pyplot as plt

def bin_data(x, y, n_bins, width_type = 'variable'):
    """
    A function for condensing one-dimensional time series data via binning.

    Parameters
    ----------
    x : array
        Time series data.
    y : array
        Corresponding rates of change [y = x']
    n_bins : int
        Total number of bins.
    width_type : string
        Determines whether the bins are of `equal` width, in which case the
        domain is split into equally sized regions, or of `variable` width,
        which entails creating bins that hold roughly equal numner of points.

    Returns
    -------
    xs : array of length n_bins containing the new states
    ys : array of length n_bins containing the new rates
    ws : array of length n_bins with bin weights
    """
    
    total = len(x)
    ws = []
    xs = []
    ys = []
    
    if width_type == 'equal':
        x_min = np.min(x)
        x_max = np.max(x)
        bin_edges = np.linspace(x_min, x_max, n_bins+1)

        for i in range(n_bins):
            mask = (x>=bin_edges[i]) & (x<=bin_edges[i+1])
            n = np.count_nonzero(mask)
            ws.append(n/total)
            xs.append((bin_edges[i]+bin_edges[i+1])/2) # centre of the bin
            if n != 0:
                ys.append(np.mean(y[mask]))
            else:
                ys.append(0)
    elif width_type == 'variable':
        x_sorted, y_sorted = np.copy(x), np.copy(y)
        y_sorted = y_sorted[x.argsort(axis = 0)].squeeze(1) # sorting xs in increasing
        x_sorted.sort(axis = 0)                             # order, ys accordingly
        x_split = np.array_split(x_sorted, n_bins)
        y_split = np.array_split(y_sorted, n_bins)
        for i in range(len(x_split)):
            xs.append(x_split[i].mean())
            ys.append(y_split[i].mean())
            ws.append(len(x_split)/total)
    else:
        raise Exception("Invalid size_type chosen. Must be 'bin_width' or 'point_n'.")
    
    xs = np.array(xs).reshape(-1,1)
    ys = np.array(ys).reshape(-1,1)
    ws = np.array(ws).reshape(-1)
    return xs, ys, ws

def time_series(func, init_state, time_step, point_n, diffusion = 0):
    """
    A function for generating time series data from a time-independent dynamical
    system x' = F(x) [ + D(x) xi'] using the explicit Euler method.

    Parameters
    ----------
    func : function
        the rate of change function F(x) whose input and output is an array of
        length d
    init_state : array of length d
        the initial state of the system
    time_step : real>0
        length of the time step dt in the explicit Euler scheme
    point_n : int>0
        number of points N in the time series (initial state included)
    diffusion : function
        the diffusion matrix D in the SDE dx = F(x) dt + D(x) d xi, where xi is
        Gaussian white noise

    Returns
    -------
    time_series : ndarray
        time series data of shape [N,d]

    """
    try:
        dim = len(init_state)
    except:
        dim = 1

    # if np.array([func(init_state)]).squeeze().shape[0] != dim:
    #     raise Exception("The dimensions of init_state and the output of func do not match.")

    try:
        D = diffusion*np.eye(dim)
    except ValueError:
        print("The argument diffusion must either be a real or a square matrix whose dimensions\
                match init_state and the output of func.")

    X = [init_state]
    for i in range(point_n):
        X_next = X[-1] + func(X[-1])*time_step \
                 + np.squeeze(np.sqrt(2)*np.dot(D, np.random.randn(D.shape[1],1)*np.sqrt(time_step)).T)
        X.append(X_next)
    time_series = np.array(X)
    return time_series

def add_noise(x, sigma = 1):
    """ Simulates measurement noise on the time series data.
    
    Parameters
    ----------
    x : array
        time series data
    sigma : float, default 1
        noise amplitude
    
    Returns
    -------
    x_noisy : array
        noisy time series, where each entry of x has been offset by a normally
        distrubuted random number of variance sigma**2
    """
    
    x_noisy = x + np.random.randn(*x.shape)*sigma
    
    return x_noisy
    

def ts_plot3(data):
    """
    Plots the time series of a 3-dimensional dynamical system.

    Parameters
    ----------
    data : ndarray
        time series data of shape [N,3]

    """
    x,y,z = data[:,0],data[:,1],data[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    N = data.shape[0]
    for i in range(N):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.viridis(i/N))
    plt.show()


def ts_scatter3(data):
    """
    Scatters the time series of a 3-dimensional dynamical system.

    Parameters
    ----------
    data : ndarray
        time series data of shape [N,3]

    """
    x,y,z = data[:,0],data[:,1],data[:,2]
    N = data.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x, y, z, color=plt.cm.viridis([i/N for i in range(N)]))
    plt.show()
