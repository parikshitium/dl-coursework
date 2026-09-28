import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary


NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=42)


plt.figure(figsize=(10,7))
plt.scatter(X_blob[:, 0],
            X_blob[:, 1],
            c=y_blob,
            cmap=plt.cm.RdYlBu)
plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_features)
        )
    def forward(self,x):
        return self.layers(x)

model = BlobModel(input_features=NUM_FEATURES,
                  output_features=NUM_CLASSES,
                  hidden_units=8).to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(params=model.parameters(),

                            lr=0.1)
def accuracy(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy


torch.manual_seed(42)

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)


for epoch in range(100):
    model.train()
    y_logits = model(X_blob_train)
    y_preds = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy(y_true=y_blob_train,
                   y_pred=y_preds)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    model.eval()

    with torch.inference_mode():
        test_logits = model(X_blob_test)
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc = accuracy(y_blob_test,
                            test_preds)
        if epoch%10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%')
model.eval()
with torch.inference_mode():
    y_logits = model(X_blob_test)
y_pred_probs = torch.softmax(y_logits,dim=1)
y_preds = y_pred_probs.argmax(dim=1)



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Training')
plot_decision_boundary(model,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title('Testing')
plot_decision_boundary(model,X_blob_test,y_blob_test)
plt.show()