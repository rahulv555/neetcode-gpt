import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pe = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for d in range(d_model):
                if(d%2==0): # d = 2*i
                    pe[pos][d] = np.sin(pos / np.power(10000, d/d_model))
                else: # d = 2i + 1
                    pe[pos][d] = np.cos(pos / np.power(10000, (d-1.0)/d_model))

        pe = np.round(pe, 5)
        return pe