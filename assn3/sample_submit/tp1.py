import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://nextjournal.com/gkoehler/pytorch-mnist
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 13, kernel_size=5)
        self.conv3 = nn.Conv2d(13, 26, kernel_size=3)
        self.conv2 = nn.Conv2d(26, 26, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(650, 100)
        # self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 26)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 4))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 650)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)


# https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
def load_dataset(addr):
    data_path = addr
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                               torchvision.transforms.ToTensor()
                             ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
  network.load_state_dict(torch.load('./results/model.pth'))
  # optimizer.load_state_dict(torch.load('./results/optimizer.pth'))
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      # print(data)
      # print(data.dtype)
      # print(data.shape)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      # print(pred, target)
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
 

n_epochs = 1
batch_size_train = 7
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.5
log_interval = 20
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)



random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# train_loader = load_dataset('../data_train/')
test_loader = load_dataset('../data_train/')

# test()
for epoch in range(1, n_epochs + 1):
  # train(epoch)
  test()