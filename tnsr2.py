import torch

torch.manual_seed(7)
tnsr1 = torch.rand(1,1,1,10)
tnsr2 = tnsr1.squeeze()
print(f'{tnsr1}\n{tnsr1.shape}')
print(f'{tnsr2}\n{tnsr2.shape}')