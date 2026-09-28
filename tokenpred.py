import torch
from torch import nn
import torch.nn.functional as F

sentence = 'joga bonito joga bonito'
tokens = sentence.split()

inputs = tokens[:-1]
targets = tokens[1:]

vocab = {
    '<UNK>': 0,
    'joga': 1,
    'bonito': 2
}

input_ids = [vocab[word] for word in inputs]
target_ids = [vocab[word] for word in targets]

input_ids = torch.tensor(input_ids)
target_ids = torch.tensor(target_ids)

embedding = nn.Embedding(3,4)
linear = nn.Linear(4,3)
optimizer = torch.optim.SGD(
    list(embedding.parameters()) + list(linear.parameters()),
    lr=0.1
)
for epoch in range(500):
    logits = linear(embedding(input_ids))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits,target_ids)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''if epoch%10==0:
        print(f'{epoch} | {loss:.4f}')'''
word = 'bonito'
word_id = torch.tensor([vocab[word]])
with torch.inference_mode():
    logits = linear(embedding(word_id))
    print(logits)
    prediction = torch.argmax(logits,dim=1)

print(prediction)