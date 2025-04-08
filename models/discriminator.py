import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.LeakyReLU(0.2)
        )
        self.downsample = nn.AvgPool1d(2) if stride == 2 else nn.Identity()
        
    def forward(self, x):
        return self.downsample(self.conv(x))

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._make_discriminator()
            for _ in range(num_scales)
        ])
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(2**(i+1)) for i in range(num_scales-1)
        ])
        
    def _make_discriminator(self):
        return nn.ModuleList([
            DiscriminatorBlock(1, 32, stride=2),
            DiscriminatorBlock(32, 64, stride=2),
            DiscriminatorBlock(64, 128, stride=2),
            DiscriminatorBlock(128, 256, stride=2),
            DiscriminatorBlock(256, 512, stride=2),
            nn.Conv1d(512, 1, 3, padding=1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, T] of audio samples
        Returns:
            list of (features, score) tuples for each scale
        """
        results = []
        
        # Process each scale
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsamplers[i-1](x)
            
            # Input must be [B, C, T]
            curr_x = x.unsqueeze(1) if x.dim() == 2 else x
            
            # Get intermediate features
            features = []
            for layer in disc[:-1]:
                curr_x = layer(curr_x)
                features.append(curr_x)
            
            # Get final score
            score = disc[-1](curr_x)
            
            results.append((features, score))
            
        return results

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([
            self._make_discriminator()
            for _ in range(len(periods))
        ])
        
    def _make_discriminator(self):
        return nn.ModuleList([
            DiscriminatorBlock(1, 32),
            DiscriminatorBlock(32, 64),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            nn.Conv1d(512, 1, 3, padding=1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Tensor [B, T] of audio samples
        Returns:
            list of (features, score) tuples for each period
        """
        results = []
        
        # Process each period
        for period, disc in zip(self.periods, self.discriminators):
            # Reshape input to [B, C, T/P, P]
            batch_size = x.size(0)
            padded_len = (x.size(-1) // period + 1) * period
            padded_x = F.pad(x, (0, padded_len - x.size(-1)))
            curr_x = rearrange(padded_x, 'b (t p) -> b 1 t p', p=period)
            
            # Get intermediate features
            features = []
            for layer in disc[:-1]:
                curr_x = layer(curr_x)
                features.append(curr_x)
            
            # Get final score
            score = disc[-1](curr_x)
            
            results.append((features, score))
            
        return results

class VocoderDiscriminator(nn.Module):
    """Combined multi-scale and multi-period discriminator"""
    def __init__(self, num_scales=3, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.msd = MultiScaleDiscriminator(num_scales=num_scales)
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        
    def forward(self, x):
        """
        Args:
            x: Tensor [B, T] of audio samples
        Returns:
            msd_out: List of (features, score) from multi-scale discriminator
            mpd_out: List of (features, score) from multi-period discriminator
        """
        return self.msd(x), self.mpd(x)
