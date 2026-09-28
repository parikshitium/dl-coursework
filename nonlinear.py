import torch
from torch import nn
import matplotlib.pyplot as plt

X = torch.arange(-10,10,1,dtype=torch.float32)

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0),x)
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1/(1+torch.exp(-x))
print(sigmoid(X))
plt.plot(torch.sigmoid(X))
plt.plot(sigmoid(X))
plt.show()