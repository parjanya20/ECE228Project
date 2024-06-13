import numpy as np
np.random.seed(0)
def create_synthetic_data(n_samples):
    g = np.random.choice([0,1,2,3],n_samples)
    p_y = [0.2,0.4,0.6,0.8]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if np.random.rand() < p_y[g[i]]:
            y[i] = 1
    X = np.zeros((n_samples,3))
    X[:,0] = np.random.normal(0,1,n_samples) + 2*y
    X[:,1] = np.random.normal(0,1,n_samples) + 2*y
    for i in range(n_samples):
        if g[i] == 0:
            if y[i] == 1:
                X[i,2] = np.random.normal(-1,1)
            else:
                X[i,2] = np.random.normal(1,1)
        elif g[i] == 1:
            if y[i] == 1:
                X[i,2] = np.random.normal(1,1)
            else:
                X[i,2] = np.random.normal(-1,1)
        elif g[i] == 2:
            if y[i] == 1:
                X[i,2] = np.random.normal(2,1)
            else:
                X[i,2] = np.random.normal(0,1)
        elif g[i] == 3:
            if y[i] == 1:
                X[i,2] = np.random.normal(-1,1)
            else:
                X[i,2] = np.random.normal(1,1)
    return X,y,g
