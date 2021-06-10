import numpy as np
import matplotlib.pyplot as plt
from dynsys import *
import time

def lorenzsys(X, sigma=10, rho=28, beta=8/3):
    x, y, z = X
    v = np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])
    return v

def customsys(state, k=0.1):
    x, y, z = state
    f = np.array([
        2*y,
        -x,
        0
    ])
    return f

dt = 0.01
T = 50
X0 = np.array([1,0,0])
X = np.array([X0])

start_time = time.time()
dynsys = DynamicalSystem(customsys);
X = dynsys.timeseries(X0, dt, int(T/dt), D = np.diag([0,0,1]))

print(time.time() - start_time)

scatter3(X)
