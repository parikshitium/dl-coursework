import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


weight = 0.7
bias = 0.3

X = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = weight*X +bias

train_split = int(.8*len(X))
X_train,y_train = X[:train_split], y[:train_split]
X_test,y_test = X[train_split:], y[train_split:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

def plot(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,10))
    plt.scatter(train_data,train_labels,c='b',s=4,label='Training Data')
    plt.scatter(test_data,test_labels,c='r',s=4,label='Test Data')

    if predictions is not None:
        plt.scatter(test_data,predictions,c='brown',s=4,label='Predictions')
    plt.legend(prop={'size':14})
    plt.show()

torch.manual_seed(42)

model1 = LinearRegressionModel()

with torch.inference_mode():
    y_pred = model1(X_test)

loss_fn = torch.nn.L1Loss()
opt = torch.optim.SGD(params=model1.parameters(),
                      lr=0.01)

torch.manual_seed(42)

epoch_count = []
loss_vals = []
test_loss_vals =[]

for epoch in range(200):
    model1.train()
    y_pred = model1(X_train)
    loss = loss_fn(y_pred,y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()
    model1.eval()

    with torch.inference_mode():
        test_pred = model1(X_test)
        test_loss = loss_fn(test_pred,y_test)

    if epoch%10 == 0:
        epoch_count.append(epoch)
        loss_vals.append(loss)
        test_loss_vals.append(test_loss)
        print(f'Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}')
        print(model1.state_dict())

plt.plot(epoch_count,np.array(torch.tensor(loss_vals).numpy()),label='Train Loss')
plt.plot(epoch_count,np.array(torch.tensor(test_loss_vals).numpy()),label='Test Loss')
plt.title('Training Loss Curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend();
plt.show()

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = 'the_first_one.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

'''print(f'Saving model to {MODEL_PATH}')'''
torch.save(obj=model1.state_dict(),
           f=MODEL_SAVE_PATH)


load_model = LinearRegressionModel()
load_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
load_model.eval()
with torch.inference_mode():
    load_preds = load_model(X_test)
print(load_preds==test_pred)