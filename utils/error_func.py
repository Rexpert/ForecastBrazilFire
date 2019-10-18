import numpy as np


def mse(y, yhat):
    return np.square(np.subtract(y, yhat)).mean()


def rmse(y, yhat):
    return np.sqrt(mse(y, yhat))


def mape(y, yhat):
    errors = np.subtract(y, yhat)
    return np.mean(np.divide(errors, y)*100)
