import sys
import pathlib

import glob
import itertools
import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from gan import *

PYTHON_DIR = pathlib.Path(sys.executable).parent.parent.resolve()
torch.ops.load_library(str(PYTHON_DIR.joinpath("lib", "libpt_ocl.so")))

from utils import ReplayBuffer

class ImageDataset():
    def __init__(self, base_dir: str | pathlib.Path, split: str = "train", transform = None) -> None:
        if isinstance(base_dir, str): base_dir = pathlib.Path(base_dir)
        self.transform = transforms.Compose(transform)
        self.file_A = list(base_dir.joinpath('{}/A/'.format(split)).glob('*.*'))
        self.file_B = list(base_dir.joinpath('{}/B/'.format(split)).glob('*.*'))
        # print(base_dir)

    def __len__(self):
        return max(len(self.file_A) , len(self.file_B))
    def __getitem__(self, idx):
        image_A = self.transform(Image.open(self.file_A[idx]))
        image_B = self.transform(Image.open(self.file_B[random.randint(0, len(self.file_B) - 1)]))
        return {
            'A': image_A,
            'B': image_B
        } 
    
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
        
#DIR = SIGNS_DIR.joinpath("64x64_SIGNS")
#train_dataset = ImageDataset(DIR, "train_signs", transform=transform)
#test_dataset = ImageDataset(DIR, "test_signs", transform=transform)
#val_dataset = ImageDataset(DIR, "val_signs", transform=transform)



epoch = 0
n_epochs = 200
batch_size = 16
lr = 2e-4
size = 256
input_nc = 3
output_nc = 3
decay_epoch = 100
n_cpu = 8

device = torch.device('privateuseone')  

base_dir = pathlib.Path(__file__).parent.parent.joinpath('datasets/summer2winter_yosemite')
print(base_dir)

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)
        
netG_A2B = GeneratorBlock(input_nc, output_nc)
netG_B2A = GeneratorBlock(input_nc, output_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(input_nc)


netG_A2B = netG_A2B.apply(fn=weights_init_normal)
netG_B2A = netG_B2A.apply(fn=weights_init_normal)
netD_A = netD_A.apply(fn=weights_init_normal)
netD_B = netD_B.apply(fn=weights_init_normal)

if str(device) != 'cpu':
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

# Pérdidas

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr = lr, betas = (0.5,0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr = lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr = lr, betas=(0.5, 0.999))

# Schedulers
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0)
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
            

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# inputs y targets
Tensor = torch.Tensor

target_real = Tensor(batch_size).fill_(1.0).to(device)
target_fake = Tensor(batch_size).fill_(0.0).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataloader
transform = [
    transforms.Resize(int(size*1.12), Image.Resampling.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]

dataloader = DataLoader(ImageDataset(base_dir, transform=transform),batch_size=batch_size, shuffle=True, num_workers=n_cpu, drop_last=True)

def Disc_GAN_loss(D2, fake2, real2, fake_2_buffer, loss, target_real, target_fake):
    pred_real = D2(real2)
    loss_D2_real = loss(pred_real, target_real)

    fake2 = fake_2_buffer.push_and_pop(fake2)
    pred_fake = D2(fake2.detach())
    loss_D2_fake = loss(pred_fake, target_fake)
    loss_D2 = (loss_D2_real + loss_D2_fake) * 0.5
    return loss_D2

def Gen_GAN_loss(G, D, real, loss, target_real):
    fake = G(real)
    pred_fake = D(fake)
    L = loss(pred_fake, target_real)
    return L, fake

def cycle_loss(G1, G2, real, loss):
    recovered = G2(G1(real))
    L = loss(recovered, real)
    return L

def identity_loss(G, real, loss):
    same = G(real)
    L = loss(same, real)
    return L

from livelossplot import PlotLosses
from utils import Logger

logger = Logger(n_epochs, len(dataloader), epoch=epoch)
liveloss= PlotLosses()

for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        optimizer_G.zero_grad()
        
        loss_GAN_A2B, fake_B = Gen_GAN_loss(netG_A2B, netD_B, real_A, criterion_GAN, target_real)
        loss_GAN_B2A, fake_A = Gen_GAN_loss(netG_B2A, netD_A, real_B, criterion_GAN, target_real)
        
        loss_cycle_ABA = cycle_loss(netG_A2B, netG_B2A, real_A, criterion_cycle)
        loss_cycle_BAB = cycle_loss(netG_B2A, netG_A2B, real_B, criterion_cycle)
        
        loss_identity_A = identity_loss(netG_B2A, real_A, criterion_identity)
        loss_identity_B = identity_loss(netG_A2B, real_B, criterion_identity)
        
        loss_G = (loss_GAN_A2B + loss_GAN_B2A) + 10.0*(loss_cycle_ABA + loss_cycle_BAB) + 5.0*(loss_identity_A + loss_identity_B)
        loss_G.backward()
        
        optimizer_G.step()
        
        # Optimizer discriminativas
        
        optimizer_D_A.zero_grad()
        
        loss_D_A = Disc_GAN_loss(netD_A, fake_A, real_A, fake_A_buffer, criterion_GAN, target_real, target_fake)
        loss_D_A.backward()
        optimizer_D_A.step()
        
        optimizer_D_B.zero_grad()
        
        loss_D_B = Disc_GAN_loss(netD_B, fake_B, real_B, fake_B_buffer, criterion_GAN, target_real, target_fake)
        loss_D_B.backward()
        optimizer_D_B.step()
        
        log_values = {
            'loss_G': loss_G,
            'loss_G_identity': loss_identity_A + loss_identity_B,
            'loss_G_GAN': loss_GAN_A2B + loss_GAN_B2A,
            'loss_G_cycle': loss_cycle_ABA + loss_cycle_BAB,
            'loss_D': loss_D_A + loss_D_B
        }
        
        logger.log(log_values, images = {'real_a': real_A, 'real_b': real_B, 'fake_a': fake_A, 'fake_b': fake_B})
    
    liveloss.update(log_values)
    liveloss.draw()
    
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
