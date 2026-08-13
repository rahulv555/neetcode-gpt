import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embeddingLayer = nn.Embedding(vocabulary_size, 16)
        self.linearLayer = nn.Linear(16, 1)
        self.sigmoidLayer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embeddings = self.embeddingLayer(x)
        meanvector = embeddings.mean(dim=1)
        z1 = self.linearLayer(meanvector)
        return torch.round(self.sigmoidLayer(z1), decimals=4)

