import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        X = np.array(X)
        w = np.zeros(X.shape[1])
        b = 0
        n = X.shape[0]
        

        for e in range(epochs):

            #forward
            y_pred = (X @ w.T) + b
            err = y_pred - y
            

            #grad
            dldw = (2/n)*(err @ X)
            dldb = (2/n)*np.sum(err)

            #update weight
            w = w - lr*dldw
            b = b - lr*dldb


        return (np.round(w, 5), round(b, 5))

        
