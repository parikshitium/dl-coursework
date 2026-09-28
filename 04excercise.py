import torch
from torch import nn
from pathlib import Path
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
train_dir = data_path / 'train'
test_dir = data_path / 'test'

train_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transformer,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transformer,
                                 target_transform=None)

train_dtload = DataLoader(dataset=train_data,
                          shuffle=True,
                          batch_size=32)
test_dtload = DataLoader(dataset=test_data,
                         shuffle=False,
                         batch_size=32)

class_names = train_data.classes
class Model(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,
                      stride=1,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_units,hidden_units,
                      kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units,hidden_units,
                      kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units,hidden_units,
                      kernel_size=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*169,output_shape)
        )
    def forward(self,x: torch.tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = Model(input_shape=3,
              hidden_units=20,
              output_shape=len(class_names)).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0,0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0,0
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc = (test_pred_labels == y).sum().item()/len(test_pred_logits)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int):
    results = {'train_loss':[],
               'train_acc':[],
               'test_loss':[],
               'test_acc':[]}
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model=model,
                                          optimizer=optimizer,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn)
        test_loss,test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn)
        #print(f'Epoch: {epoch+1}\ntrain_loss: {train_loss:.4f}\ntrain_acc: {train_acc:.4f}\ntest_loss: {test_loss:.4f}\ntest_acc: {test_acc:.4f}')
        results['train_loss'].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results['train_acc'].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results['test_loss'].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    return results



results = train(model=model,
      train_dataloader=train_dtload,
      test_dataloader=test_dtload,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=20)

def plotandloss(results: dict[str, list[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

plotandloss(results=results)