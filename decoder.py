import torch
from torch import nn
from torch.nn import functional as F
from attetion import SelfAtteniton

class VAE_AttentionBlock(nn.Sequential):

    def __init__(self, channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)
        self.attetion = SelfAtteniton(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, feature, height, width)

        residue = x

        n, c, h, w = x.shape

        x = x.view(n, c, h*w)

        x = x.transpose(-1,-2)

        x = self.attetion(x)

        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)
        
        x+=residue

        return x




class VAE_ResidualBlock(nn.Sequential):
    
    def __init__(self, in_channel, out_channel):

        super().__init_()

        self.groupnorm_1 = nn.GroupNorm(32, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channe, out_channel, kernel_size = 1, padding = 0)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channel, height, width)

        residue = x

        x = self.groupnorm_1(x)

        x= F.silu()

        x= self.conv_1(x)

        x= self.groupnorm_2(x)

        x= Fv.silu()

        x= self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),

            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(512,512),

            VAE_AttentionBlock(512), 

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.Upsample(scale_factor = 2), 

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            nn.Upsample(scale_factor = 2), 

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor = 2),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.Silu(), 

            nn.Conv2d(128, 3, kernel_size = 3, padding = 1)

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (batch_size, 4, height/8, width/8)
        x /= 0.18215

        for module in self:
            x= module(x)

        # (batch_size, 3, height, width)
        return x 

