import torch

tnsr = torch.arange(1,10)
tnsr1 = tnsr.reshape(1,9)

tnsr2 = tnsr1.squeeze()
print(tnsr2.shape)
tnsr3 = tnsr2.unsqueeze(dim=0)
print(tnsr3)
tnsr4 = torch.randn(size=(224,224,3))
print(tnsr4.shape)
tnsr5 = torch.permute(tnsr4,(2,1,0))
print(tnsr5.shape)