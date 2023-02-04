
# from config import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image


import sys
import pathlib

PYTHON_DIR = pathlib.Path(sys.executable).parent.parent.resolve()
torch.ops.load_library(str(PYTHON_DIR.joinpath("lib", "libpt_ocl.so")))


class Net(nn.Module):
    def __init__(self, num_channels) -> None:
        super(Net, self).__init__()

        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        self.fc1 = nn.Linear(self.num_channels*4*8*8, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.num_channels*4, 6)

    def forward(self, x): # (3, 64, 64)
        x = self.conv1(x) # (n, 64, 64)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2)) # (n, 32, 32)
        x = self.conv2(x) # (2*n, 32, 32)
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(x, 2)) # (2*n, 16, 16)
        x = self.conv3(x) # (2*n, 16, 16)
        x = self.bn3(x)
        x = F.relu(F.max_pool2d(x, 2)) # (4*n, 8, 8)

        # flatten
        x = x.view(-1, self.num_channels*4*8*8)

        # fc
        x = self.fc1(x)
        x = self.fcbn1(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.8, training=True)
        x = self.fc2(x)
        
        # softmax
        x = F.log_softmax(x, dim=1)
        
        return x


class SIGNSDataset():
    def __init__(self, base_dir: str | pathlib.Path, split: str = "train", transform = None) -> None:
        if isinstance(base_dir, str): base_dir = pathlib.Path(base_dir)
        path = base_dir.joinpath(split)
        files = path.iterdir()

        self.filenames = list(filter(lambda x: str(x).endswith(".jpg"), files))
        self.labels = [int(f.name[0]) for f in self.filenames]
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class RunningMetric():
    def __init__(self) -> None:
        self.S = 0
        self.N = 0
    
    def update(self, val, size):
        self.S += val
        self.N += size

    def __call__(self):
        try:
            return self.S / float(self.N)
        except ZeroDivisionError as e:
            return 0

# transform = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#     ]
# )

DIR = pathlib.Path("/home/rml/dev/pytorch/datasets/SIGNS").joinpath("64x64_SIGNS")
print(DIR)
train_dataset = SIGNSDataset(DIR, "train_signs", transform=transforms.ToTensor())
test_dataset = SIGNSDataset(DIR, "test_signs", transform=transforms.ToTensor())
val_dataset = SIGNSDataset(DIR, "val_signs", transform=transforms.ToTensor())

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


device = torch.device('privateuseone')
net = Net(32).to(device)
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

num_epochs = 25
for epoch in range(1, num_epochs + 1):
    print("Epoch {} / {}".format(epoch, num_epochs))
    print("-"*10)

    running_loss = RunningMetric()
    running_ac  = RunningMetric()
    running_loss_v = RunningMetric()
    running_ac_v  = RunningMetric()

    for phase in ('train', 'val'):
        if phase == "train":
            net.train()
        else:
            net.eval()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            outputs1 = outputs.to('cpu')
            _, preds = torch.max(outputs1, 1)
            # preds = np.max(outputs1)
            loss = loss_fn(outputs, targets)
            if phase == 'train':
                loss.backward()
                optimizer.step()

            batch_size = inputs.size()[0]
            running_loss.update(batch_size*loss.item(),
                                batch_size)
            running_ac.update(torch.sum(preds == targets.to('cpu')),
                                batch_size)
            print("\r {} Loss: {:4f} - Accuracy {:4f}".format(phase, running_loss(), running_ac()), end="")
        print()
        
        



