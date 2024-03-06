import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.l2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        # with torch.no_grad():
        x = self.l1(x)
        # with torch.no_grad():
        x = self.l2(x)
        return x

net = Net()
optim = torch.optim.SGD(net.parameters(), lr=0.001)
critereon = torch.nn.MSELoss()
for i in range(1000):
    optim.zero_grad()
    label = torch.tensor([1.])
    x = torch.tensor([1.])
    y = net(x)
    loss = critereon(y, label)
    loss.backward()
    optim.step()
    print('loss: ', loss.item(), '     y: ', y.item())