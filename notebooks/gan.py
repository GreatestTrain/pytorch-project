import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        ]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        return x + self.conv_block(x)
    
    
class GeneratorBlock(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9) -> None:
        super(GeneratorBlock, self).__init__()
        input_nc = int(input_nc)
        output_nc = int(output_nc)
        n_residual_blocks = int(n_residual_blocks)
        
        # Conv Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        
        in_features = 64
        out_features = in_features * 2
        
        # Encoder
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            
            in_features = out_features
            out_features *= 2
            
        # Residual Block
        for _ in range(n_residual_blocks):
            model += [
                ResidualBlock(in_features)
            ]
            
        # Decoder
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            
            in_features = out_features
            out_features = in_features // 2
            
        # Salida
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]
    
        self.model = nn.Sequential(*model)
        
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    "PatchGAN: Discrimina estilo o textura."
    
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ] 
        
        model += [
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ] 
        
        model += [
            nn.Conv2d(512, 1, 4, padding=1),
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], - 1)