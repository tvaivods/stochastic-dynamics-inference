import numpy as np
import matplotlib.pyplot as plt

class DynamicalSystem:
    """The DynamicalSystem class is intended to simplify data generation from a dynamical system."""
    def __init__(self, func):
        """Sets the functional form of the dynamical system."""
        self.func = func
    def timeseries(self, X0, dt, steps, D = 0):
        """Generates a time-series of N points starting from state X0 using time-step size dt. The diffusion matrix can be specified as the argument D."""
        self.N = steps + 1
        self.dt = dt
        self.D = D*np.eye(3)
        X = [X0]
        for i in range(steps):
            x_next = X[-1] + self.func(X[-1])*dt + \
                     np.squeeze(np.dot(self.D, np.random.randn(3,1)*np.sqrt(dt)).T)
            X.append(x_next)
        self.data = np.array(X)
        return self.data

def plot3(X):
    x,y,z = X[:,0],X[:,1],X[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    N = X.shape[0]
    for i in range(N):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.viridis(i/N))
    plt.show()
def scatter3(X):
    x,y,z = X[:,0],X[:,1],X[:,2]
    N = X.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x, y, z, color=plt.cm.viridis([i/N for i in range(N)]))
    plt.show()
