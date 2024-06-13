import numpy as np
np.random.seed(0)

def estimate_test_py(y_tr_true,y_tr_pred,qy_hat):
    n_classes = len(np.unique(y_tr_true))
    py_y_hat = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            py_y_hat[i,j] = np.mean((y_tr_true == j)&(y_tr_pred == i)) + 1e-6
    py = np.sum(py_y_hat,axis=0)
    qy = np.dot(np.diag(py),np.dot(np.linalg.inv(py_y_hat),qy_hat))
    qy = qy+1e-6
    qy = qy / np.sum(qy)
    #If values of qy are outside 0 and 1 clip them to 0 and 1 as done in BBSE paper
    qy = np.clip(qy,0.01,0.99)

    return qy, py


def recalibrate(p_yx,qy,py):
    qy = qy/py
    return ((p_yx*qy).T/np.sum(p_yx*qy,axis=1)).T
