import torch
import torch.nn as nn

# --- 1. MODEL ARCHITECTURE ---
# You MUST ensure these match your notebook exactly
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        q = self.query(x).view(batch, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h*w)
        v = self.value(x).view(batch, -1, h*w)
        attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, c, h, w)
        return self.gamma * out + x

class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Module 2)
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        # Attention Bottleneck (Module 5)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), SelfAttention(256))

        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.final = nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = torch.relu(self.dec1(b) + e2) # Skip Connection
        d2 = torch.relu(self.dec2(d1) + e1) # Skip Connection
        return torch.tanh(self.final(d2))