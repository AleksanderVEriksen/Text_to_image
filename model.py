import torch
import torch.nn as nn
from torch.amp import autocast

def double_conv(inn, out):
    conv = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out, out, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(1, emb_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, t):
        # t: (batch_size,) or (batch_size, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = self.linear1(t.float())
        x = self.act(x)
        x = self.linear2(x)
        return x  # (batch_size, emb_dim)

class UNET(nn.Module):
    # UNET for 3 channel images (RGB)
    def __init__(self):
        super(UNET, self).__init__()
        # Downsampling
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        #self.down_conv3 = double_conv(128, 256)
        #self.down_conv4 = double_conv(256, 512)
        #self.down_conv5 = double_conv(512, 1024)

        # Time embedding
        self.time_mlp = TimeEmbedding(128)
        
        # Upsampling
        #self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        #self.up_conv1 = double_conv(1024, 512)
        #self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        #self.up_conv2 = double_conv(512, 256)
        #self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        #self.up_conv3 = double_conv(256, 128)
        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)
        self.out = nn.Conv2d(64, 2, kernel_size=1)
    
    @autocast(device_type='cuda')
    def forward(self, x, t):
        x1 = self.down_conv1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        #x4 = self.max_pool_2x2(x3)
        #x5 = self.down_conv3(x4)
        #x6 = self.max_pool_2x2(x5)
        #x7 = self.down_conv4(x6)
        #x8 = self.max_pool_2x2(x7)
        #x9 = self.down_conv5(x8)
        # Add time embedding
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]  # Reshape for broadcasting
        x3 = x3 + t_emb

        # decode
        #x = self.up_trans1(x9)
        #y = crop_tensor(x7, x)
        #x = self.up_conv1(torch.cat([x, y], dim=1))
        #x = self.up_trans2(x)
        #y = crop_tensor(x5, x)
        #x = self.up_conv2(torch.cat([x, y], dim=1))
        #x = self.up_trans3(x)
        #y = crop_tensor(x3, x)
        #x = self.up_conv3(torch.cat([x, y], dim=1))
        x = self.up_trans4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv4(torch.cat([x, y], dim=1))
        x = self.out(x)
        print(x.size())        

        return x


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
                h.append(x) # Storing output for skip connection
                x = self.downscale(x) # Downscale ready for the next layer
            
        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
                x = self.upscale(x) # Upscale
                x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function
            
        return x









if __name__ == "__main__":
    model = UNET()
    Image = torch.randn((1, 3, 572, 572))
    t = torch.tensor([0])  # Example timestep
    print(model(Image, t))
