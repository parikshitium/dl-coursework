import torch
from torch import nn

sentence = 'cat dog cat'

tokens = sentence.split()
vocab = {
    'cat': 0,
    'dog': 1,
    'car': 2
}

token_id = [vocab[token] for token in tokens]

embedding = nn.Embedding(3,4)

x = torch.tensor([0,1])

target = torch.tensor([1,0])
linear = nn.Linear(4,3)
loss_fn = nn.CrossEntropyLoss()
scores = linear(embedding(x))
loss = loss_fn(scores, target)
optimizer = torch.optim.SGD(
    list(embedding.parameters()) + list(linear.parameters()),
    lr=0.1
)

for epoch in range(100):
    optimizer.zero_grad()
    scores = linear(embedding(x))
    loss = loss_fn(scores,target)
    loss.backward()
    optimizer.step()

    if epoch%10==0:
        print(f'Epoch: {epoch} | Loss: {loss.item():.4f}')