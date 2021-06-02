import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dynsys import DynamicalSystem
import time

def lorenzsys(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    v = np.array([
        sigma*(y - x),
        x*(rho - z) - y,
        x*y - beta*z
    ])
    return v

def customsys(state, k=0.1):
    x, y, z = state
    f = np.array([
        y,
        -x,
        0
    ])
    return f

dt = 0.01
T = 50
X0 = np.array([0,1,1])
X = np.array([X0])

start_time = time.time()
dynsys = DynamicalSystem(customsys);
X = dynsys.timeseries(X0, dt, int(T/dt), D = np.diag([0,0,10]))

print(time.time() - start_time)
help(DynamicalSystem.timeseries)

dynsys.plot3()
