import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_linear = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784) (28x28 = 784)
        # Return the model's prediction to 4 decimal places
        x = self.first_linear(images)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return torch.round(x, decimals=4)



