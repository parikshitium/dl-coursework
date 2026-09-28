import torch
from torch import nn
import torch.nn.functional as F


embedding = nn.Embedding(3,4)
x = torch.tensor([0,1,2])
optimizer = torch.optim.SGD(embedding.parameters(),
                            lr=0.1)
def trainloop(embedding,
              x,
              optimizer,
              id1,
              id2,
              target):
    vector = embedding(x)
    similarity = F.cosine_similarity(vector[id1], vector[id2], dim=0)
    loss = (similarity - target)**2
    print(f'Before:\n{similarity} | {loss:.4f}')
    
    print(F.cosine_similarity(vector[0],vector[1],dim=0))
    print(F.cosine_similarity(vector[0],vector[2],dim=0))
    for epoch in range(100):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        vector = embedding(x)
        similarity = F.cosine_similarity(vector[id1], vector[id2], dim=0)
        loss = (similarity - target)**2
        if epoch%10==0:
            print(f'{epoch} | {similarity} | {loss:.4f}')
            
    print(F.cosine_similarity(vector[0],vector[1],dim=0))
    print(F.cosine_similarity(vector[0],vector[2],dim=0))

trainloop(embedding=embedding,
          x=x,
          optimizer=optimizer,
          id1=0,
          id2=1,
          target=1)
trainloop(embedding=embedding,
          x=x,
          optimizer=optimizer,
          id1=0,
          id2=2,
          target=0)