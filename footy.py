from datasets import load_dataset
import re
import torch
from torch import nn
from torch.utils.data import TensorDataset,random_split,DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset('SoccerNet/SN-echoes','whisper_v3')

texts = dataset['en']['text'][:10000]
text = ' '.join(texts)

words = re.findall(f"[a-z]+", text.lower())

vocab = sorted(set(words))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i:word for word,i in word_to_id.items()}
ids = [word_to_id[word] for word in words]

context_size = 4
inputs = []
targets = []
for i in range(len(ids) - context_size):
    inputs.append(ids[i:i + context_size])
    targets.append(ids[i+context_size])

in_tnsr = torch.tensor(inputs)
targ_tnsr = torch.tensor(targets)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Embedding(4787,32),
            nn.Flatten(),
            nn.Linear(32*context_size,4787)
        )
    def forward(self,x):
        x = self.layer(x)
        return x

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
dataset = TensorDataset(in_tnsr,targ_tnsr)
train_size = int(0.9*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_dtload = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True)
test_dtload = DataLoader(dataset=test_dataset,
                         batch_size=64)


for epoch in range(10):
    model.train()
    for batch, (x_batch, y_batch) in enumerate(train_dtload):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = loss_fn(output,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss = []
    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in test_dtload:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_fn(output,y_batch)
            test_loss.append(loss.item())
    avg = sum(test_loss)/len(test_loss)
    correct = 0
    total = 0
    with torch.inference_mode():
        for x_batch, y_batch in test_dtload:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
        accuracy = correct/total

temp = 0.5
top_k = 10
context = ['the','team','has','won']
model.eval()

for i in range(10):
    context_ids = [word_to_id[word] for word in context] 
    x = torch.tensor(context_ids).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(x)
        probabilities = torch.softmax(output/temp,dim=1)
        top_probs, top_ids = torch.topk(probabilities,top_k)
        top_probs = top_probs/top_probs.sum()
        prediction_idx = torch.multinomial(top_probs,1).item()
        prediction_id = top_ids[0,prediction_idx].item()
    prediction = id_to_word[prediction_id]
    print(prediction)
    context = context[1:] + [prediction]