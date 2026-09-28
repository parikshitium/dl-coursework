import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
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


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss,train_acc = 0,0
    model.to(device)

    for batch, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_preds = model(X)
        loss = loss_fn(y_preds,y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_preds.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /=  len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%')


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
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
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_preds.argmax(dim=1)
                                    )
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')
def time(start: float,
         end: float,
         device: torch.device = device):
    total_time = end - start
    print(f'Time Taken: {total_time}')
    return total_time



class FashionModelV1(nn.Module):
    def __init__(self,input_shape: int,hidden_units: int,output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    def forward(self,x):
        return self.layers(x)

torch.manual_seed(42)
model1 = FashionModelV1(input_shape=784,
                     hidden_units=10,
                     output_shape=len(fields)
                     )
model = model1
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.1)
loss_fn = nn.CrossEntropyLoss()
torch.manual_seed(42)
start_time = timer() 

for epoch in tqdm(range(3)):
    tqdm.write(f'Epoch: {epoch}')
    train_step(data_loader=train_dtload,
               model=model,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
            )
    test_step(data_loader=test_dtload,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
    )
end_time = timer()

model1_time = time(start_time,end_time)

class FashionModelV2(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape),
        )
    def forward(self,x):
        return self.layers(x)

    
torch.manual_seed(42)
model2 = FashionModelV2(input_shape=784,
                       hidden_units=10,
                       output_shape=len(fields)).to(device)
model = model2
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.1)
torch.manual_seed(42)
start_time = timer()

for epoch in tqdm(range(3)):
    tqdm.write(f'Epoch: {epoch}')
    train_step(data_loader=train_dtload,
               loss_fn=loss_fn,
               model=model,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dtload,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
    )
end_time = timer()
model2_time = time(start_time,end_time)


class FashionModelV3(nn.Module):
    def __init__(self,
                 input_shape:int,
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
                      kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7,output_shape)
        )
    def forward(self,x: torch.Tensor):
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model3 = FashionModelV3(input_shape=1,
                       hidden_units=10,
                       output_shape=len(fields)).to(device)

model = model3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1) 


start = timer()
torch.manual_seed(42)

for epoch in tqdm(range(3)):
    tqdm.write(f'Epoch: {epoch}')
    train_step(data_loader=train_dtload,
               model=model,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
               )
    test_step(data_loader=test_dtload,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
              )
end = timer()

model3_time = time(start=start,end=end)

torch.manual_seed(42)
def model_eval(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
    loss,acc=0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            y_preds = model(X)
            loss += loss_fn(y_preds,y)
            acc += accuracy_fn(y,y_preds.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item(),
            'model_accuracy': acc}


model1_results = model_eval(model=model1,
                 data_loader=test_dtload,
                 loss_fn=loss_fn,
                 accuracy_fn=accuracy_fn,
                 device=device)
model2_results = model_eval(model=model2,
                            data_loader=test_dtload,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn,
                            device=device)
model3_results = model_eval(model=model3,
                            data_loader=test_dtload,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn,
                            device=device)

compare_results = pd.DataFrame([model1_results,model2_results,model3_results])
compare_results['training_time'] = [model1_time,model2_time,model3_time]
compare_results.set_index('model_name')['model_accuracy'].plot(kind='barh')
'''plt.xlabel('accuracy')
plt.ylabel('model');
plt.show()'''

def make_predictions(model: torch.nn.Module,
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
random.seed(42)
test_samples = []
test_labels =[]

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
pred_probs = make_predictions(model=model3,
                              data=test_samples)
pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9,9))
nrows=3
ncols=3

'''for i, sample in enumerate(test_samples):
    plt.subplot(nrows,ncols,i+1)
    plt.imshow(sample.squeeze(),cmap='gray')
    pred_label = fields[pred_classes[i]]
    truth_label = fields[test_labels[i]]
    title_text = {f'Pred: {pred_label} | Truth: {truth_label}'}
    if pred_label == truth_label:
        plt.title(title_text,fontsize=10,c='g')
    else:
        plt.title(title_text,fontsize=10,c='r')
    plt.axis(False);
plt.show()'''

y_preds = []
model2.eval()
for X,y in tqdm(test_dtload, desc='Making predictions...'):
    X,y = X.to(device), y.to(device)
    y_logit = model2(X)
    y_pred = torch.softmax(y_logit,dim=0).argmax(dim=1)
    y_preds.append(y_pred.cpu())
y_pred_tensor = torch.cat(y_preds)
confmat = ConfusionMatrix(num_classes=len(fields), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=fields,
                                figsize=(10,7));
path = Path('models')
path.mkdir(parents=True,
           exist_ok=True)
name = 'cnn.pth'
save_path = path/name
torch.save(obj=model3.state_dict(),
           f=save_path)