import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
from timeit import default_timer as timer
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


device = 'cuda'
train_data = datasets.MNIST(
    train=True,
    download=True,
    root='data',
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.MNIST(
    train=False,
    download=True,
    root='data',
    transform=ToTensor(),
    target_transform=None
)

fields = train_data.classes

train_dtload = DataLoader(dataset=train_data,
                          batch_size=32,
                          shuffle=True)
test_dtload = DataLoader(dataset=test_data,
                         batch_size=32,
                         shuffle=False)

class Model(nn.Module):
    def __init__(self,input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7,output_shape)
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = Model(input_shape=1,
              hidden_units=10,
              output_shape=len(fields)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

accuracy = accuracy_fn

def time(start: float,
         end: float,
         device: torch.device = device):
    total_time = end - start
    return total_time

def train_step(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device,
               ):
    train_loss,train_acc=0,0
    model.to(device)

    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        loss = loss_fn(y_preds,y)
        train_loss += loss
        train_acc += accuracy_fn(y,
                                 y_preds.argmax(dim=1)) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%')
def test_step(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    
    test_loss,test_acc = 0,0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            test_preds = model(X)
            test_loss += loss_fn(test_preds,y)
            test_acc += accuracy_fn(y,
                                    test_preds.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

start = timer()
for epoch in tqdm(range(3)):
    tqdm.write(f'Epoch: {epoch}')
    train_step(data_loader=train_dtload,
               model=model,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               optimizer=optimizer,
               device=device)
    test_step(data_loader=test_dtload,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
end = timer()
time_taken = time(start=start,
                  end=end,
                  device=device)
print(time_taken)
def make_predictions(model: nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)


test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)
pred_probs = make_predictions(model=model,
                              data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

y_preds = []
model.eval()
for X,y in tqdm(test_dtload, desc='making preds'):
    X,y = X.to(device), y.to(device)
    y_logit = model(X)
    y_pred = torch.softmax(y_logit,dim=0).argmax(dim=1)
    y_preds.append(y_pred.cpu())
y_pred_tensor = torch.cat(y_preds)
confmat = ConfusionMatrix(num_classes=len(fields),task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=fields,
                                figsize=(10,7))
plt.show()