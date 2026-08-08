import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        if training:
            mu = np.mean(x, axis=0)
            var = np.mean(np.power(x-mu, 2))

            running_mean = np.round(np.multiply(1-momentum,running_mean) +np.multiply(momentum,mu), 4)
            running_var = np.round(np.multiply(1-momentum,running_var) + np.multiply(momentum,var), 4)

            x = (x-mu)/np.sqrt(var + eps)

            x = gamma*x + beta
            x = np.round(x, 4)
            return (x, running_mean, running_var)
        else:
            x = np.round((x-np.array(running_mean))/np.sqrt(np.array(running_var) + eps), 4)
            x = gamma*x + beta
            return (x, running_mean, running_var)


