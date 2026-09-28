from sklearn.datasets import make_circles
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

num = 1000
X, y = make_circles(num,
                    noise=.03,
                    random_state=42)

circles = pd.DataFrame({'X1':X[:,0],
                        'X2':X[:,1],
                        'Label':y})

plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu);

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linearlayers = nn.Sequential(
            nn.Linear(in_features=2,out_features=10),
            nn.Linear(in_features=10,out_features=10),
            nn.Linear(in_features=10,out_features=1)
        )
    def forward(self,x):
        return self.linearlayers(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model = CircleModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(params=model.parameters(),
    lr=0.1)
def accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy
#model.eval()
with torch.inference_mode():
    y_logits = model(X_test.to(device))
    y_preds_probs = torch.sigmoid(y_logits)
    y_preds = torch.round(y_preds_probs)[:5]
    y_preds_label = torch.round(torch.sigmoid(model(X_test.to(device))[:5]))

torch.cuda.manual_seed(42)

X_train,y_train = X_train.to(device), y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)

for epoch in range(1000):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits,
                   y_train)
    acc = accuracy(y_true=y_train,
                   y_pred=y_pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()

    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy(y_true=y_test,y_pred=test_pred)
    if epoch%100==0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%')

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model,X_train,y_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model,X_test,y_test)
plt.show()