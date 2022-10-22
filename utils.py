import numpy as np

def RMSE(y_trues, y_preds):
    return np.sqrt(np.mean((y_trues - y_preds) ** 2))


def MCRMSE(y_trues, y_preds):
    return np.mean(np.sqrt(np.mean((y_trues - y_preds.T) ** 2, axis=1)))

