import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x)  #(input_dim, ) - 1D array
        W1 = np.array(W1) #(hidden_dim, input_dim)  = (r, c)
        b1 = np.array(b1) #(hidden_dim, )
        W2 = np.array(W2) #(output_dim, hidden_dum)
        b2 = np.array(b2) #(output_dim,)
        y_true = np.array(y_true) #(output_dim,)

        
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, W2.T) + b2
        y_pred = z2

        loss = np.mean(np.power(y_pred-y_true, 2))

        #dL/dw2
        dLdz2 = (2*(y_pred-y_true))/len(y_true)
        dLdw2 = np.outer(dLdz2, a1) #a1 = dz2 / dw2

        #dL/db2
        dLdb2 = dLdz2 # * 1 : 1 = dz2 / db2

        #dL/dw1
        da1dz1 = (z1 > 0).astype(np.float64)
        dLda1 = np.dot(dLdz2,W2)
        dLdz1 = dLda1 * da1dz1 
        dLdw1 = np.outer(dLdz1, x)#W2 = dz2/da1, x = dz1/dw1

        #dL/db1
        dLdb1 = dLdz1
        
        return {'loss':np.round(loss,4), 'dW1':np.round(dLdw1,4), 'db1':np.round(dLdb1,4), 'dW2':np.round(dLdw2,4), 'db2':np.round(dLdb2, 4)}
    




        
