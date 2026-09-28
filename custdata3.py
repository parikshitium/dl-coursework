import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import datasets,transforms
from custdata import plot_transformed_images
import os
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import requests

device = 'cuda' if torch.cuda.is_available else 'cpu'
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
train_dir = data_path / 'train'
test_dir = data_path / 'test'

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.classes

train_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor()
])

image_path_list = list(data_path.glob('*/*/*.jpg'))

simple_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)
train_dtload_simple = DataLoader(dataset=train_data_simple,
                                 batch_size=32,
                                 shuffle=True)
test_dtload_simple = DataLoader(dataset=test_data_simple,
                                batch_size=32,
                                shuffle=False)
class Model(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        ) 
    def forward(self,x: torch.Tensor):
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x

torch.manual_seed(42)
modelI = Model(input_shape=3,
            hidden_units=10,
            output_shape=len(class_names)).to(device)

image_batch, label_batch = next(iter(train_dtload_simple))

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
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
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
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
        return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int):
    results = {'train_loss':[],
               'train_acc':[],
               'test_loss':[],
               'test_acc':[]}

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer)
        test_loss,test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn)
        '''print(
            f'Epoch: {epoch+1} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Train Accuracy: {train_acc:.4f} | '
            f'Test Loss: {test_loss:.4f} | '
            f'Test Accuracy: {test_acc:.4f}'
            )'''

        results['train_loss'].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results['train_acc'].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results['test_loss'].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results
torch.manual_seed(42)
torch.cuda.manual_seed(42)

modelII = Model(input_shape=3,
                hidden_units=10,
                output_shape=len(train_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=modelII.parameters(),
                             lr=0.001)

start_m1 = timer()

model1_results = train(model=modelII,
                      train_dataloader=train_dtload_simple,
                      test_dataloader=test_dtload_simple,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=5)
end_m1 = timer()
total_time_m1 = end_m1 - start_m1
#print(f'Training time: {total_time_m1} seconds')

def plot_loss_curve(results: dict[str, list[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

train_data_trivial = datasets.ImageFolder(train_dir,transform=train_transform_trivial)
test_data_trivial = datasets.ImageFolder(test_dir,transform=test_transform)

torch.manual_seed(42)
train_dtload_trivial = DataLoader(train_data_trivial,
                                  batch_size=32,
                                  shuffle=True)
test_dtload_trivial = DataLoader(test_data_trivial,
                                 batch_size=32,
                                 shuffle=False)
torch.manual_seed(42)

modelIII = Model(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_trivial.classes)
).to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=modelIII.parameters(),
                             lr=0.001)

start_m2 = timer()
model2_results = train(model=modelIII,
                       train_dataloader=train_dtload_trivial,
                       test_dataloader=test_dtload_simple,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5)
end_m2 = timer()
total_time_m2 = end_m2 - start_m2

model1 = pd.DataFrame(model1_results)
model2 = pd.DataFrame(model2_results)
def plot_4():
    plt.figure(figsize=(15,10))
    epochs = range(len(model1))

    plt.subplot(2,2,1)
    plt.plot(epochs,model1['train_loss'],label='Model 1')
    plt.plot(epochs,model2['train_loss'],label='Model 2')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(epochs,model1['test_loss'],label='Model 1')
    plt.plot(epochs,model2['test_loss'],label='Model 2')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(epochs,model1['train_acc'],label='Model 1')
    plt.plot(epochs,model2['train_acc'],label='Model 2')
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(epochs,model1['test_acc'],label='Model 1')
    plt.plot(epochs,model2['test_acc'],label='Model 2')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

custom_image_path = data_path / '04-pizza-dad.jpeg'
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64,64))
])


def pred_and_plot(model: torch.nn.Module,
                  image_path: str,
                  class_names: list[str] = None,
                  transform = None,
                  device: torch.device = device):
    target_image = torchvision.io.read_image(str(image_path))
    target_image = target_image/255

    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()

    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))
    target_image_pred_prob = torch.softmax(target_image_pred,dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_prob,dim=1)

    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        title = f'Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_prob.max().cpu():.3f}'
    else:
        title = f'Pred: {target_image_pred_label} | Prob: {target_image_pred_prob.max().cpu():.3f}'
    plt.title(title)
    plt.axis(False)
    plt.show()
pred_and_plot(model=modelIII,
              device=device,
              transform=custom_image_transform,
              class_names=class_names,
              image_path=custom_image_path)