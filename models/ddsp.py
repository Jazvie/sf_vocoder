import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class HarmonicOscillator(nn.Module):
    """Differentiable harmonic oscillator for source generation"""
    def __init__(self, sample_rate=16000, frame_rate=200, n_harmonics=100):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.n_harmonics = n_harmonics
        self.frame_size = sample_rate // frame_rate  # 80 samples at 16kHz/200Hz
        
    def forward(self, f0, harmonic_dist, phase_shift=None):
        """
        Args:
            f0: Tensor [B, 1, T] of fundamental frequencies in Hz at frame_rate
            harmonic_dist: Tensor [B, H, T] of harmonic amplitudes at frame_rate
            phase_shift: Optional tensor [B, 1, T] of phase offsets at frame_rate
        Returns:
            signal: Tensor [B, T * frame_size] of audio samples at sample_rate
        """
        batch_size = f0.shape[0]
        n_frames = f0.shape[-1]
        device = f0.device
        
        # Create time axis for full audio resolution
        t = torch.arange(n_frames * self.frame_size, device=device).float()
        t = t.view(1, -1).expand(batch_size, -1)
        
        # Upsample control signals to audio rate using linear interpolation
        f0 = F.interpolate(f0, size=n_frames * self.frame_size, mode='linear', align_corners=True)
        harmonic_dist = F.interpolate(harmonic_dist, size=n_frames * self.frame_size, mode='linear', align_corners=True)
        
        if phase_shift is not None:
            phase_shift = F.interpolate(phase_shift, size=n_frames * self.frame_size, mode='linear', align_corners=True)
        
        # Initialize signal
        signal = torch.zeros(batch_size, n_frames * self.frame_size, device=device)
        
        # Generate harmonics
        for harm_idx in range(self.n_harmonics):
            # Frequency for this harmonic
            freq = f0 * (harm_idx + 1)
            
            # Phase accumulation
            phase = 2 * np.pi * freq * t / self.sample_rate
            
            if phase_shift is not None:
                phase = phase + phase_shift.squeeze(1)
            
            # Add harmonic
            harmonic = torch.sin(phase) * harmonic_dist[:, harm_idx]
            signal = signal + harmonic
        
        return signal

class FilterModule(nn.Module):
    """Learnable filter module"""
    def __init__(self, n_filter_coeffs=512, frame_rate=200, sample_rate=16000):
        super().__init__()
        self.n_filter_coeffs = n_filter_coeffs
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.frame_size = sample_rate // frame_rate
        
    def forward(self, source, filter_coeffs):
        """
        Args:
            source: Tensor [B, T*frame_size] of source signal at sample_rate
            filter_coeffs: Tensor [B, N, T] of filter coefficients at frame_rate
        Returns:
            filtered: Tensor [B, T*frame_size] of filtered signal at sample_rate
        """
        batch_size = source.shape[0]
        n_frames = filter_coeffs.shape[-1]
        
        # Upsample filter coefficients to audio rate
        filter_coeffs = F.interpolate(
            filter_coeffs,
            size=n_frames * self.frame_size,
            mode='linear',
            align_corners=True
        )
        
        # Apply filtering using overlap-add
        hop_size = self.frame_size // 2
        window = torch.hann_window(self.frame_size, device=source.device)
        
        # Pad source for overlap-add
        source = F.pad(source, (self.frame_size, self.frame_size))
        
        # Initialize output buffer
        filtered = torch.zeros_like(source)
        
        # Process each frame
        for i in range(0, source.shape[-1] - self.frame_size, hop_size):
            # Extract frame
            frame = source[:, i:i + self.frame_size] * window
            
            # Get filter coefficients for this frame
            current_filter = filter_coeffs[:, :, i//hop_size]
            
            # Apply filter
            frame_filtered = F.conv1d(
                frame.unsqueeze(1),
                current_filter.unsqueeze(1),
                padding=self.n_filter_coeffs//2
            ).squeeze(1)
            
            # Add to output buffer with overlap-add
            filtered[:, i:i + self.frame_size] += frame_filtered * window
        
        # Remove padding and normalize
        filtered = filtered[:, self.frame_size:-self.frame_size]
        
        return filtered

class DDSPSynthesizer(nn.Module):
    """Complete DDSP-based source-filter synthesizer"""
    def __init__(self, sample_rate=16000, frame_rate=200, n_harmonics=100, n_filter_coeffs=512):
        super().__init__()
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.frame_size = sample_rate // frame_rate  # 80 samples at 16kHz/200Hz
        
        self.oscillator = HarmonicOscillator(
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            n_harmonics=n_harmonics
        )
        self.filter = FilterModule(
            n_filter_coeffs=n_filter_coeffs,
            frame_rate=frame_rate,
            sample_rate=sample_rate
        )
        
    def forward(self, f0, harmonic_dist, filter_coeffs, noise_amp=None, phase_shift=None):
        """
        Args:
            f0: Tensor [B, 1, T] of fundamental frequencies at frame_rate
            harmonic_dist: Tensor [B, H, T] of harmonic amplitudes at frame_rate
            filter_coeffs: Tensor [B, N, T] of filter coefficients at frame_rate
            noise_amp: Optional tensor [B, 1, T] of noise amplitudes at frame_rate
            phase_shift: Optional tensor [B, 1, T] of phase shifts at frame_rate
        Returns:
            signal: Tensor [B, T*frame_size] of synthesized audio at sample_rate
        """
        # Generate source signal
        source = self.oscillator(f0, harmonic_dist, phase_shift)
        
        # Add noise if provided
        if noise_amp is not None:
            # Upsample noise amplitude to audio rate
            noise_amp = F.interpolate(
                noise_amp,
                size=source.shape[-1],
                mode='linear',
                align_corners=True
            )
            noise = torch.randn_like(source) * noise_amp.squeeze(1)
            source = source + noise
        
        # Apply filter
        signal = self.filter(source, filter_coeffs)
        
        return signal
