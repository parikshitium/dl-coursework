import torch
from torch import nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_samples = 1000

X,y = make_moons(n_samples=n_samples,
                 noise=0.03,
                 random_state=42
                 )

'''plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdBu)
plt.show()'''

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=42)

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=1)
        )
    def forward(self,x):
        return self.layers(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = MoonModel().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)
def accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy
torch.cuda.manual_seed(42)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(1000):
    model.train()
    y_logits = model(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)
    acc = accuracy(y_train,y_preds)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy(y_test,test_preds)

        if epoch%10==0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%')


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Training')
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1,2,2)
plt.title('Testing')
plot_decision_boundary(model,X_test,y_test)
plt.show()
