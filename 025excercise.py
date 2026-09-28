import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchmetrics import Accuracy
from helper_functions import plot_decision_boundary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
N = 100 
D = 2 
K = 3 
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8') 
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) 
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
'''plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()'''

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
class SpiralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=3)
        )
    def forward(self,x):
        return self.layers(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = SpiralModel().to(device)
acc_fn = Accuracy(task='multiclass',num_classes=3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params=model.parameters(),
                             lr=0.02)
torch.cuda.manual_seed(42)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(1000):
    model.train()
    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits,y_train)
    acc = acc_fn(y_preds,y_train)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    model.eval()

    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,y_test)
        test_acc = acc_fn(test_preds,y_test)

        if epoch%100==0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}')

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Training')
plot_decision_boundary(model,X_train,y_train)
plt.subplot(1,2,2)
plt.title('Testing')
plot_decision_boundary(model,X_test,y_test)
plt.show()