import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = datasets.MNIST(
    train=True,
    download=True,
    root='data',
    transform=ToTensor()
)
train_data = Subset(train_data,list(range(6000)))
test_data = datasets.MNIST(
    train=False,
    download=True,
    root='data',
    transform=ToTensor()
)

class SiameseData(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        image1, label1 = self.data[index]
        same = random.choice([True,False])
        while True:
            index2 = random.randrange(len(self.data))
            image2, label2 = self.data[index2]
            if same and label2 == label1:
                break
            if not same and label2 != label1:
                break
        target = int(same)
        return image1, image2, target

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*196,32)
        )
    def forward(self,x):
        x = self.layer2(self.layer1(x))
        return x

encoder = Encoder()

class Siamese(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self,image1,image2):
        embed1 = self.encoder(image1)
        embed2 = self.encoder(image2)
        return F.cosine_similarity(embed1, embed2, dim=1)

model = Siamese(encoder).to(device)

train_dtload = DataLoader(
    dataset=SiameseData(train_data),
    shuffle=True,
    batch_size=128
)
test_dtload = DataLoader(
    dataset=SiameseData(test_data),
    batch_size=128
)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.1)

for epoch in range(2):
    epoch_loss = 0
    model.train()
    for image1,image2,target in train_dtload:
        image1,image2,target = image1.to(device),image2.to(device),target.to(device)
        similarity = model(image1,image2)
        batch_loss = ((target - similarity)**2).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    epoch_loss = epoch_loss/len(train_dtload)

model.eval()
total_correct = 0
with torch.inference_mode():
    for image1,image2,target in test_dtload:
        image1,image2,target = image1.to(device),image2.to(device),target.to(device)
        test_sim = model(image1,image2)
        pred = (test_sim >= 0.5).int()
        correct = pred == target
        total_correct += correct.int().sum()
accuracy = (total_correct.item()/len(test_dtload.dataset))*100
print(accuracy)