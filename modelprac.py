import torch
from torch import nn
import matplotlib.pyplot as plt


weight = 123
bias = 22
X = torch.arange(0,1456,3).unsqueeze(dim=1)
y = weight*X +bias

train_split = int(.75*len(X))
X_train,y_train = X[:train_split],y[:train_split]
X_test,y_test = X[train_split:],y[train_split:]

class LinearRegressonModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weight*x + self.bias

def plot(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,predictions=None):
    plt.figure(figsize=(10,10))
    plt.scatter(train_data,train_label,c='b',s='4',label='Training Data')
    plt.scatter(test_data,test_label,c='r',s='4',label='Test Data')

    if predictions is not None:
        plt.scatter(predictions,c='g',s='4',label='Predictions')
    plt.legend()
    plt.show()

torch.manual_seed(42)
model = LinearRegressonModelV2()

with torch.inference_mode():
    y_pred = model(X_test)
loss_fn = torch.nn.L1Loss()
opt = torch.optim.SGD(params=model.parameters(),
                      lr=(.1))

torch.manual_seed(42)
epoch_count = []
loss_vals = []
test_loss_vals = []

for epoch in range(100):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()

    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test)

    if epoch%10 == 0:
        epoch_count.append(epoch)
        loss_vals.append(loss.item())
        test_loss_vals.append(test_loss.item())


plt.plot(epoch_count,loss_vals,label='Train Loss')
plt.plot(epoch_count,test_loss_vals,label='Test Loss')
plt.title('Traning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()