import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor):
        # timesteps: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb  # (B, dim)

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
    def __init__(self, in_channels=3, out_channels=3, num_classes=10, embedding_dim=512, label_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.label_dropout = label_dropout

        self.time_embed = SinusoidalTimeEmbedding(embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.null_label_embedding = nn.Parameter(torch.zeros(embedding_dim))
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.label_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Projection for fused embedding -> bottleneck channels (512)
        self.fuse_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, 512)
        )

        # Downsampling
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(in_channels, 32)
        self.down_conv2 = double_conv(32, 64)
        self.down_conv3 = double_conv(64, 128)
        self.down_conv4 = double_conv(128, 256)
        self.down_conv5 = double_conv(256, 512)

        # Upsampling
        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(128, 64)
        self.up_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _forward_core(self, x, fused_embed):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)
        # Inject fused embedding
        x9 = x9 + fused_embed[:, :, None, None]
        # Decoder (skip order)
        x = self.up_trans1(x9)
        x = self.up_conv1(torch.cat([x, crop_tensor(x7, x)], 1))
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, crop_tensor(x5, x)], 1))
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, crop_tensor(x3, x)], 1))
        x = self.up_trans4(x)
        x = self.up_conv4(torch.cat([x, crop_tensor(x1, x)], 1))
        x = self.out(x)
        return x

    @autocast(device_type='cuda')
    def forward(self, x, timesteps, labels=None, guidance_scale=None):
        # Time embedding
        t_sin = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_sin)  # (B, D)

        # Training-time label dropout (classifier-free)
        if labels is not None and self.training:
            keep = (torch.rand_like(labels.float()) > self.label_dropout)
            effective_labels = labels.clone()
            effective_labels[~keep] = -1
        else:
            effective_labels = labels

        def label_encode(lab):
            if lab is None:
                return self.label_mlp(self.null_label_embedding.expand(x.size(0), -1))
            if (lab == -1).all():
                return self.label_mlp(self.null_label_embedding.expand(x.size(0), -1))
            emb = self.label_embedding(torch.clamp(lab, min=0))
            return self.label_mlp(emb)

        # Conditional path
        cond_label_emb = label_encode(effective_labels) if labels is not None else label_encode(None)
        cond_fused = self.fuse_proj(t_emb + cond_label_emb)  # (B, 512)

        # If no guidance scale, single forward
        if guidance_scale is None or labels is None:
            return self._forward_core(x, cond_fused)

        # Unconditional path (all null labels, no dropout)
        null_label_emb = label_encode(torch.full_like(labels, -1))
        null_fused = self.fuse_proj(t_emb + null_label_emb)

        # Two passes
        x_cond = self._forward_core(x.clone(), cond_fused)
        x_null = self._forward_core(x, null_fused)

        # Classifier-free guidance combination
        return x_null + guidance_scale * (x_cond - x_null)


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
