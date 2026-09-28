import torch
from torch import nn

vocab = {
    'cat': 0,
    'dog': 1,
    'car': 2
}

embedding = nn.Embedding(3,2)
target = torch.tensor([1.,1.])
output = embedding(torch.tensor([0]))
loss_fn = nn.MSELoss()
loss = loss_fn(output.squeeze(),target)
optimizer = torch.optim.SGD(
    embedding.parameters(),
    lr=0.1
)

print(f'Before: {embedding.weight[0]}')
for i in range(10):
    output = embedding(torch.tensor([0]))
    loss = loss_fn(output.squeeze(), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Step {i}: {loss.item():.4f}')

print(f'After: {embedding.weight[0]}')