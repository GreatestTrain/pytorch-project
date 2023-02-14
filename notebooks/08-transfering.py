# %%
# from config import *

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

import pathlib
from PIL import Image

import sys
import pathlib

PYTHON_DIR = pathlib.Path(sys.executable).parent.parent.resolve()
torch.ops.load_library(str(PYTHON_DIR.joinpath("lib", "libpt_ocl.so")))

device = torch.device('privateuseone')
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg = vgg.to(device)

for params in vgg.parameters():
    params.requires_grad = False

# %%
last_sequential_layer = list(vgg.children())[-1]
*list_of_layers, last_layer = list(last_sequential_layer)
in_features = last_layer.in_features

# %%
vgg.fc = nn.Linear(in_features, 6)
vgg.fc.requires_grad_ = True
vgg.classifier = nn.Sequential(*(list_of_layers + [vgg.fc]))

# %%
transform = transforms.Compose(
  [transforms.RandomHorizontalFlip(), #data augmentation
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
  ]
)

# %%
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

DIR = pathlib.Path("/home/rml/dev/pytorch/datasets/SIGNS").joinpath("64x64_SIGNS")
print(DIR)
train_dataset = SIGNSDataset(DIR, "train_signs", transform=transforms.ToTensor())
test_dataset = SIGNSDataset(DIR, "test_signs", transform=transforms.ToTensor())
val_dataset = SIGNSDataset(DIR, "val_signs", transform=transforms.ToTensor())

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# %%
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(vgg.parameters(),lr=1e-3, momentum=0.9)

# %%
def train_eval(model: nn.Module, optimizer: optim.Optimizer, loss_fn , dataloaders: list, epochs: int = 10, lr: float = 0.01):
    for g in optimizer.param_groups:
        g['lr'] = lr
    
    train_loader, val_loader = dataloaders

    for epoch in range(1, epochs + 1):
        print("Epoch {} / {}".format(epoch, epochs))
        print("-"*10)

        running_loss = RunningMetric()
        running_ac  = RunningMetric()
        running_loss_v = RunningMetric()
        running_ac_v  = RunningMetric()

        for phase in ('train', 'val'):
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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

train_eval(vgg, optimizer=optimizer, loss_fn=loss_fn, dataloaders=[train_loader, val_loader], epochs=10, lr=1e-3)
