import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn 
from timeit import default_timer as timer
from tqdm.auto import tqdm


device = 'cpu'

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
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
indx = train_data.class_to_idx
image,label = train_data[0]
'''plt.imshow(image.squeeze())
plt.title(fields[label])
plt.show()'''

#torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows,cols = 4,4

for i in range(1,rows*cols+1):
    random_idx = torch.randint(0,len(train_data),size=[1]).item()
    img, label = train_data[random_idx]
    '''fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(),cmap='gray')
    plt.title(fields[label])
    plt.axis(False);
'''
train_dtload = DataLoader(dataset=train_data,
                    batch_size=32,
                    shuffle=True)

test_dtload = DataLoader(dataset=test_data,
                         batch_size=32,
                         shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dtload))
#torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch), size=[1]).item()
img ,label = train_features_batch[random_idx], train_labels_batch[random_idx]
fig.add_subplot(rows,cols,i)
plt.imshow(img.squeeze(),cmap='gray')
plt.title(fields[label])


class FashionModel(nn.Module):
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
model = FashionModel(input_shape=784,
                     hidden_units=10,
                     output_shape=len(fields))
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

def time(start: float,
         end: float,
         device: torch.device = None):
    total_time = end - start
    print(f'Time Taken: {total_time:.3f}')
    return total_time

torch.manual_seed(42)
start_time = timer()

for epoch in tqdm(range(3)):
    print(f'Epoch: {epoch}')
    train_loss = 0

    for batch, (X,y) in enumerate(train_dtload):
        model.train()
        y_preds = model(X)
        loss = loss_fn(y_preds,y)
        train_loss += loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        model.eval()

        if batch%400 == 0:
            print(f'Looked at {batch*len(X)}/{len(train_dtload.dataset)} samples.')
    train_loss /= len(train_dtload)

    test_loss,test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dtload:
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds,y_test)
            test_acc += accuracy_fn(y_test,test_preds.argmax(dim=1))
        test_loss /= len(test_dtload)
        test_acc /= len(test_dtload)
    print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')
end_time = timer()
print(time(start_time,end_time,device=str(next(model.parameters()).device)))

torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            y_preds = model(X)
            loss += loss_fn(y_preds,y)
            acc += accuracy_fn(y,y_preds.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item(),
            'model_accuracy': acc}

model_results = eval_model(model=model,data_loader=test_dtload,loss_fn=loss_fn,accuracy_fn=accuracy_fn)
print(model_results)