import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.functional as F

def double_conv(inn, out):
    conv = nn.Sequential(
        nn.Conv2d(inn, out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out, out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(enc_feat, x):
    """
    Matcher størrelsen på enc_feat til x.
    - Hvis enc_feat er større -> crop
    - Hvis enc_feat er mindre -> pad
    """
    _, _, H, W = x.shape
    enc_H, enc_W = enc_feat.shape[2], enc_feat.shape[3]

    # *Crop hvis enc_feat er større
    if enc_H > H or enc_W > W:
        enc_feat = enc_feat[:, :, :H, :W]

    # *Pad hvis enc_feat er mindre
    if enc_H < H or enc_W < W:
        diffY = H - enc_H
        diffX = W - enc_W
        enc_feat = F.pad(enc_feat, [0, diffX, 0, diffY])  # *[left, right, top, bottom]

    return enc_feat

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(1, emb_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, t):
        # *t: (batch_size,) or (batch_size, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = self.linear1(t.float())
        x = self.act(x)
        x = self.linear2(x)
        return x  # (batch_size, emb_dim)

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, labels):
        # *labels: (batch_size,)
        x = self.embedding(labels)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x  # (batch_size, emb_dim)

class UNET(nn.Module):
    # *UNET for 2 channel images (RGB)
    def __init__(self, in_channels=3, out_channels=3, num_classes: int = 10):
        assert in_channels == out_channels, "Input and output channels must be the same"
        super(UNET, self).__init__()
        # Downsampling
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(in_channels, 32)
        self.down_conv2 = double_conv(32, 64)
        self.down_conv3 = double_conv(64, 128)
        self.down_conv4 = double_conv(128, 256)
        self.down_conv5 = double_conv(256, 512)
        #self.down_conv6 = double_conv(512, 1024)

        # *Time embedding
        self.time_mlp = TimeEmbedding(512)
        # *Label embedding (optional conditioning)
        self.label_mlp = LabelEmbedding(num_classes, 512)
        
        # Upsampling
        #self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        #self.up_conv1 = double_conv(1024, 512)
        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(128, 64)
        self.up_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    @autocast(device_type='cuda')
    def forward(self, x, t, labels=None):
        # encode
        x1 = self.down_conv1(x) # 32
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2) # 64
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4) # 128
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6) # 256
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8) # 512

        # *Add time embedding
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]  # Reshape for broadcasting
        x9 = x9 + t_emb
        # *Add label embedding if provided
        if labels is not None:
            # normalize labels input
            if isinstance(labels, (list, tuple)):
                labels = torch.tensor(labels, dtype=torch.long, device=t.device)
            elif isinstance(labels, int):
                labels = torch.tensor([labels], dtype=torch.long, device=t.device)
            elif isinstance(labels, str):
                raise TypeError("labels is a string; expected tensor/list/tuple of ints.")
            elif isinstance(labels, torch.Tensor):
                labels = labels.to(t.device).long()
            else:
                raise TypeError(f"Unsupported labels type: {type(labels)}")
            l_emb = self.label_mlp(labels)
            l_emb = l_emb[:, :, None, None]
            x9 = x9 + l_emb

        # decode
        x = self.up_trans1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv1(torch.cat([x, y], dim=1))

        x = self.up_trans2(x7)
        y = crop_tensor(x5, x)
        x = self.up_conv2(torch.cat([x, y], dim=1))

        x = self.up_trans3(x5)
        y = crop_tensor(x3, x)
        x = self.up_conv3(torch.cat([x, y], dim=1))

        x = self.up_trans4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv4(torch.cat([x, y], dim=1))

        x = self.out(x)
        return x


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1, TEST=False, num_classes: int = 10):
        super().__init__()

        self.TEST = TEST # * For MNIST dataset

        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])

        self.time_mlp = TimeEmbedding(64)
        # optional label embedding (small dim for BasicUNet)
        self.label_mlp = LabelEmbedding(num_classes, 64)

        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    def forward(self, x, t, labels=None):
        h = []
        # First the down layers
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            if i < 2: # For all but the third (final) down layer:
                h.append(x) # Storing output for skip connection
                x = self.downscale(x) # Downscale ready for the next layer
        # Add time embedding
        # * Only for testing on MNIST
        if self.TEST:
            print("Before t_emb:", x.shape)
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]  # Reshape for broadcasting
        x = x + t_emb
        # Add label embedding if provided
        if labels is not None:
            l_emb = self.label_mlp(labels.to(t.device))
            l_emb = l_emb[:, :, None, None]
            x = x + l_emb
        if self.TEST:
            print("After t_emb", x.shape)
            self.TEST = False 

        # Now the up layers
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
