import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

weight = 0.3
bias = 0.9
X = torch.arange(0,1.5,0.05)
y = weight*X + bias

train_split = int(0.8*len(X))
X_train,y_train = X[:train_split],y[:train_split]
X_test,y_test = X[train_split:],y[train_split:]

def plot(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,10))
    plt.scatter(train_data,train_labels,c='b',s=15,label='Training Data')
    plt.scatter(test_data,test_labels,c='r',s=15,label='Test Data')
    if predictions is not None:
        plt.scatter(test_data.detach(),predictions.detach(),c='g',s=15,label='Predictions')
    plt.legend()
    plt.show()

class ExcerciseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.weight*x +self.bias

torch.manual_seed(42)
model = ExcerciseModel()
loss_fn = nn.L1Loss()
opt = torch.optim.SGD(params=model.parameters(),
                      lr=0.01)

for epoch in range(300):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch%20==0:
        model.eval()

        with torch.inference_mode():
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test,y_test)
plot(predictions=y_pred_test)

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME = 'excercise.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)