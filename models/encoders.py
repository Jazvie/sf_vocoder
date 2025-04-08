import torch
import torch.nn as nn
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.LayerNorm([out_channels]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)

class ArticulatoryEncoder(nn.Module):
    """Encodes EMA data into filter parameters"""
    def __init__(self, hidden_dim=256, n_filter_coeffs=512):
        super().__init__()
        self.ema_dim = 12  # Fixed EMA dimension
        
        # Project EMA
        self.ema_proj = nn.Conv1d(self.ema_dim, hidden_dim, 1)
        
        # Dilated convolutions for temporal context
        self.convs = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, dilation=1),
            ConvBlock(hidden_dim, hidden_dim, dilation=2),
            ConvBlock(hidden_dim, hidden_dim, dilation=4),
            ConvBlock(hidden_dim, hidden_dim, dilation=8),
        ])
        
        # Project to filter coefficients
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, n_filter_coeffs, 1),
            nn.Tanh()  # Bound filter coefficients
        )
        
    def forward(self, ema):
        """
        Args:
            ema: Tensor of shape [B, 12, T] - batch of EMA sequences at 200 Hz
        Returns:
            filter_coeffs: Tensor of shape [B, N, T] - filter coefficients
        """
        # Project EMA
        x = self.ema_proj(ema)
        
        # Apply dilated convs
        for conv in self.convs:
            x = x + conv(x)  # Residual connections
            
        # Get filter coefficients
        filter_coeffs = self.output_proj(x)
        
        return filter_coeffs

class AcousticEncoder(nn.Module):
    """Encodes F0 and acoustic features into source parameters"""
    def __init__(self, hidden_dim=256, n_harmonics=100):
        super().__init__()
        
        # Input projections
        self.f0_proj = nn.Conv1d(1, hidden_dim//3, 1)
        self.loudness_proj = nn.Conv1d(1, hidden_dim//3, 1)
        self.periodicity_proj = nn.Conv1d(1, hidden_dim//3, 1)
        
        # Merge features
        self.merge = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        self.convs = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, dilation=1),
            ConvBlock(hidden_dim, hidden_dim, dilation=2),
            ConvBlock(hidden_dim, hidden_dim, dilation=4),
        ])
        
        # Outputs:
        # 1. Harmonic amplitudes for oscillator
        # 2. Noise amplitude
        # 3. Phase shift
        self.harmonic_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, n_harmonics, 1),
            nn.Softplus()
        )
        self.noise_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        self.phase_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Tanh()  # [-1, 1] phase shift
        )
        
    def forward(self, f0, loudness, periodicity):
        """
        Args:
            f0: Tensor [B, 1, T] - fundamental frequency at 200 Hz
            loudness: Tensor [B, 1, T] - loudness at 200 Hz
            periodicity: Tensor [B, 1, T] - periodicity at 200 Hz
        Returns:
            harmonic_distribution: Tensor [B, H, T] - harmonic amplitudes
            noise_amp: Tensor [B, 1, T] - noise amplitude
            phase_shift: Tensor [B, 1, T] - phase shift
        """
        # Project inputs
        f0_enc = self.f0_proj(f0)
        loudness_enc = self.loudness_proj(loudness)
        periodicity_enc = self.periodicity_proj(periodicity)
        
        # Concatenate acoustic features
        x = torch.cat([f0_enc, loudness_enc, periodicity_enc], dim=1)
        x = self.merge(x)
        
        # Apply convs
        for conv in self.convs:
            x = x + conv(x)
        
        # Get source parameters
        harmonic_dist = self.harmonic_proj(x)
        noise_amp = self.noise_proj(x)
        phase_shift = self.phase_proj(x)
        
        return harmonic_dist, noise_amp, phase_shift
