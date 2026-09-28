import torch
from torch import nn
from helper_functions import plot_predictions
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight = 0.7
bias = 0.3

X = torch.arange(0,1,0.01).unsqueeze(dim=1)
y = weight*X + bias

train_split = int(0.8*len(X))
X_train,y_train = X[:train_split],y[:train_split]
X_test,y_test = X[train_split:],y[train_split:]

torch.manual_seed(42)
torch.cuda.manual_seed(42)


model = nn.Sequential(
    nn.Linear(in_features=1,out_features=10),
    nn.Linear(in_features=10,out_features=10),
    nn.Linear(in_features=10,out_features=1)
).to(device)

loss_fn = nn.L1Loss()
opt = torch.optim.SGD(model.parameters(),
                      lr=0.01)


X_train,y_train = X_train.to(device), y_train.to(device)
X_test,y_test = X_test.to(device), y_test.to(device)

for epoch in range(1000):
    y_preds = model(X_train)
    loss = loss_fn(y_preds,y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test)
        if epoch%100 == 0:
            print(f'Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}')
model.eval()
with torch.inference_mode():
    y_preds = model(X_test)
plot_predictions(X_train.cpu(),y_train.cpu(),X_test.cpu(),y_test.cpu(),y_preds.cpu())
plt.show()