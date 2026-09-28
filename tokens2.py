import torch
from torch import nn
import torch.nn.functional as F


sentence = 'joga bonito joga!'
sentence2 = 'joga! bellisimo'
token = sentence.split()
token2 = sentence2.split()
token = [word.replace('!','') for word in token]
token2 = [word.replace('!','') for word in token2]
vocab = {'<UNK>':0}
for index, word in enumerate(token):
    if word not in vocab:
        vocab[word] = len(vocab)

token_ids = [vocab[word] for word in token]
token_ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in token2]
token_ids = torch.tensor(token_ids)

embedding = nn.Embedding(3,4)
optimizer = torch.optim.SGD(embedding.parameters(),
                            lr=0.1)
pair1 = torch.tensor([1,2])
pair2 = torch.tensor([1,0])
for epoch in range(100):
    vec1 = embedding(pair1)
    vec2 = embedding(pair2)
    sim1 = F.cosine_similarity(vec1[0],vec1[1],dim=0)
    sim2 = F.cosine_similarity(vec2[0],vec2[1],dim=0)
    loss1 = (1-sim1)**2
    loss2 = torch.clamp(sim2,min=0)**2
    loss = loss1+loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''if epoch%10==0:
        print(f'{epoch} | {sim1:.4f} | {sim2:.4f} | {loss:.4f}')'''

print(embedding.weight)