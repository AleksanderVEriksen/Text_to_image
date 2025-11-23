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
    def __init__(self, in_channels=1, base_channels=64, num_classes=10,
                 time_emb_dim=256, label_emb_dim=128, fuse_dim=512, label_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.label_dropout = label_dropout
        # --- time embedding ---
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim),
        )
        # --- label embedding ---
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        self.null_label_embedding = nn.Parameter(torch.zeros(label_emb_dim))
        nn.init.normal_(self.null_label_embedding, std=0.02)
        self.label_mlp = nn.Sequential(
            nn.Linear(label_emb_dim, fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim),
        )
        # --- fuse projection (shared) ---
        self.fuse_proj = nn.Linear(fuse_dim, fuse_dim)

        # Encoder blocks
        self.down_conv1 = self._block(in_channels, base_channels)
        self.down_conv2 = self._block(base_channels, base_channels * 2)
        self.down_conv3 = self._block(base_channels * 2, base_channels * 4)
        self.down_conv4 = self._block(base_channels * 4, base_channels * 8)
        self.down_conv5 = self._block(base_channels * 8, base_channels * 8)

        self.max_pool_2x2 = nn.MaxPool2d(2)

        # Decoder blocks
        self.up_trans1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, 2)
        self.up_conv1 = self._block(base_channels * 16, base_channels * 8)
        self.up_trans2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.up_conv2 = self._block(base_channels * 8, base_channels * 4)
        self.up_trans3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.up_conv3 = self._block(base_channels * 4, base_channels * 2)
        self.up_trans4 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.up_conv4 = self._block(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        # Per-level projection of fused embedding to channel dims
        ch = base_channels
        self.emb_proj1 = nn.Linear(fuse_dim, ch)
        self.emb_proj2 = nn.Linear(fuse_dim, ch * 2)
        self.emb_proj3 = nn.Linear(fuse_dim, ch * 4)
        self.emb_proj4 = nn.Linear(fuse_dim, ch * 8)
        self.emb_proj5 = nn.Linear(fuse_dim, ch * 8)
        self.up_emb_proj1 = nn.Linear(fuse_dim, ch * 8)
        self.up_emb_proj2 = nn.Linear(fuse_dim, ch * 4)
        self.up_emb_proj3 = nn.Linear(fuse_dim, ch * 2)
        self.up_emb_proj4 = nn.Linear(fuse_dim, ch)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU()
        )

    def _inject(self, x, emb_vec, proj):
        return x + proj(emb_vec)[:, :, None, None]

    def _forward_core(self, x, fused):
        # Encoder with injections
        x1 = self._inject(self.down_conv1(x), fused, self.emb_proj1)
        x2 = self._inject(self.down_conv2(self.max_pool_2x2(x1)), fused, self.emb_proj2)
        x3 = self._inject(self.down_conv3(self.max_pool_2x2(x2)), fused, self.emb_proj3)
        x4 = self._inject(self.down_conv4(self.max_pool_2x2(x3)), fused, self.emb_proj4)
        x5 = self._inject(self.down_conv5(self.max_pool_2x2(x4)), fused, self.emb_proj5)

        # Decoder with injections
        x = self.up_trans1(x5)
        x = torch.cat([x, crop_tensor(x4, x)], 1)
        x = self._inject(self.up_conv1(x), fused, self.up_emb_proj1)

        x = self.up_trans2(x)
        x = torch.cat([x, crop_tensor(x3, x)], 1)
        x = self._inject(self.up_conv2(x), fused, self.up_emb_proj2)

        x = self.up_trans3(x)
        x = torch.cat([x, crop_tensor(x2, x)], 1)
        x = self._inject(self.up_conv3(x), fused, self.up_emb_proj3)

        x = self.up_trans4(x)
        x = torch.cat([x, crop_tensor(x1, x)], 1)
        x = self._inject(self.up_conv4(x), fused, self.up_emb_proj4)

        return self.out(x)

    @autocast(device_type='cuda', enabled=True)
    def forward(self, x, timesteps, labels=None, guidance_scale=None):
        # Assertions
        assert x.shape[2] == 32 and x.shape[3] == 32, f"Expect 32x32 inputs, got {x.shape}"
        t_emb = self.time_mlp(self.time_embed(timesteps))  # (B, fuse_dim)

        # Classifier-free dropout
        if labels is not None and self.training and self.label_dropout > 0:
            drop_mask = torch.rand_like(labels.float()) < self.label_dropout
            labels_cf = labels.clone()
            labels_cf[drop_mask] = -1
        else:
            labels_cf = labels

        def encode_labels(lab):
            if lab is None or (lab == -1).all():
                null = self.null_label_embedding.expand(x.size(0), -1)
                return self.label_mlp(null)
            emb = self.label_embedding(torch.clamp(lab, min=0))
            return self.label_mlp(emb)

        cond_lab_emb = encode_labels(labels_cf if labels is not None else None)
        cond_fused = self.fuse_proj(t_emb + cond_lab_emb)

        if guidance_scale is None or labels is None:
            return self._forward_core(x, cond_fused)

        # Unconditional path
        null_lab_emb = encode_labels(torch.full_like(labels, -1))
        null_fused = self.fuse_proj(t_emb + null_lab_emb)
        eps_cond = self._forward_core(x.clone(), cond_fused)
        eps_null = self._forward_core(x, null_fused)
        return eps_null + guidance_scale * (eps_cond - eps_null)


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
