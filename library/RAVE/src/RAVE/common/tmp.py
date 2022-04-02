import torch

x = 20*torch.ones((32, 2))
y = torch.zeros((32))

y[16:] = 1

soft = torch.nn.LogSoftmax(dim=1)
criterion = torch.nn.NLLLoss()

loss = criterion(soft(x), y.long())

print(loss.item())
