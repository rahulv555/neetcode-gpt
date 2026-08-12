import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        
        
        sentences = positive + negative
        vocab = {}
        i = 1.0;
        words = [w for s in sentences for w in s.split(' ')]
        words.sort()

        #Mapping words to token_ids
        for w in words:
            if not vocab.get(w):
                vocab[w]=i
                i+=1
        
        encodings = []
        for s in sentences:
            encodings.append(torch.tensor([vocab[w] for w in s.split(' ')]))

        #padding each row to make them equal length
        encodings = nn.utils.rnn.pad_sequence(encodings, padding_value=0, batch_first=True)
        return encodings
