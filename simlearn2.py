import torch
from torch import nn
import torch.nn.functional as F

pairs = [(0,1),(0,2),(1,2)]
targets = [1,0,0]

embedding = nn.Embedding(3,4)
x = torch.tensor([0,1,2])
optimizer = torch.optim.SGD(embedding.parameters(),
                            lr=0.1)
for epoch in range(100):
    for pair, target in zip(pairs,targets):
        margin = 0.5
        id1, id2 = pair
        vectors = embedding(x)
        vector1 = vectors[id1]
        vector2 = vectors[id2]
        similarity = F.cosine_similarity(vector1,vector2,dim=0)
        loss = (1-similarity)**2 if target == 1 else torch.clamp(similarity - margin,min=0)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(f'{pair} | {similarity:.4f} | {loss:.4f}')