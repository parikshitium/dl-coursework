import torch
from torch import nn
from timeit import default_timer as timer
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from compvision2 import train_step,test_step,model_eval,time
from tqdm.auto import tqdm


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
            nn.Linear(hidden_units*0,output_shape)
        )
    def forward(self,x: torch.Tensor):
        x = self.block1(x)
        print(x.shape)
        x = self.block2(x)
        print(x.shape)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = FashionModelV3(input_shape=1,
                       hidden_units=10,
                       output_shape=len(fields)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)
torch.manual_seed(42)

start_time = timer()

for epoch in range(3):
    #print(f'Epoch: {epoch}')
    train_step(data_loader=train_data,
               model=model,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device
        )
    test_step(data_loader=test_data,
              model=model,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device
        )
end_time = timer()
'''print(time(start=start_time,
           end=end_time,
           device=device))   '''