# Copyright (c) 2026 M2M-TSE Implementation
# SPDX-License-Identifier: Apache-2.0
#
# Reference:
#   D. Choi and J. Choi, "Multichannel-to-Multichannel Target Sound Extraction 
#   Using Direction and Timestamp Clues", ICASSP 2025.
#   (Based on DeFTAN-II architecture)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.feature.speech import STFT, iSTFT
from wesep.modules.common.norm import select_norm


class TwoDimSplitDenseBlock(nn.Module):
    """2D Split Dense Block (2D SDB) for extracting spatial features 
    and learning local spectral-temporal relations.
    
    Modified from DenseNet for multichannel processing.
    """
    
    def __init__(self, in_channels, growth_rate=32, bn_size=4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class ClueEmbedding(nn.Module):
    """Spatio-temporal clue embedding for DoA and timestamps.
    
    Supports two encoding schemes:
    - One-hot: 360-degree resolution for discrete directions
    - Cyclic positional encoding: continuous direction encoding
    """
    
    def __init__(self, clue_dim=360, embed_dim=128, encoding_type='one-hot'):
        super().__init__()
        self.encoding_type = encoding_type
        self.embed_dim = embed_dim
        
        if encoding_type == 'one-hot':
            # One-hot encoding for direction (0-360 degrees, 1 degree resolution)
            self.direction_embed = nn.Linear(clue_dim, embed_dim)
        elif encoding_type == 'cyclic':
            # Cyclic positional encoding for continuous direction
            self.clue_proj = nn.Linear(2, embed_dim)  # sin/cos encoding
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # Timestamp embedding (temporal activity)
        self.timestamp_embed = nn.Linear(1, embed_dim)
        
    def forward(self, direction, timestamp=None):
        """
        Args:
            direction: (B,) direction in degrees [0, 360)
            timestamp: (B, T) optional temporal activity mask
        Returns:
            clue_embed: (B, C, T) where C = embed_dim
        """
        B = direction.shape[0]
        T = 1  # Default temporal dimension
        
        if timestamp is not None:
            T = timestamp.shape[1]
        
        if self.encoding_type == 'one-hot':
            # Convert direction to one-hot vector
            direction_idx = direction.long() % 360
            one_hot = F.one_hot(direction_idx, num_classes=360).float()  # (B, 360)
            direction_embed = self.direction_embed(one_hot)  # (B, embed_dim)
        else:
            # Cyclic encoding
            angle_rad = direction.float() * np.pi / 180.0
            cyclic_enc = torch.stack([
                torch.sin(angle_rad),
                torch.cos(angle_rad)
            ], dim=-1)  # (B, 2)
            direction_embed = self.clue_proj(cyclic_enc)  # (B, embed_dim)
        
        # Expand to (B, embed_dim, T)
        direction_embed = direction_embed.unsqueeze(-1).expand(-1, -1, T)
        
        if timestamp is not None:
            # Timestamp embedding
            ts_embed = self.timestamp_embed(timestamp.unsqueeze(-1))  # (B, T, embed_dim)
            ts_embed = ts_embed.permute(0, 2, 1)  # (B, embed_dim, T)
            # Combine direction and timestamp
            clue_embed = direction_embed + ts_embed
        else:
            clue_embed = direction_embed
            
        return clue_embed


class FrequencyTransformer(nn.Module):
    """F-Transformer: Analyzes relationships in spectral sequences."""
    
    def __init__(self, d_model, nhead=4, dim_feedforward=512, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, d_model, F, T) -> (B, T, d_model, F) -> (B*T, F, d_model)
        B, d_model, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, d_model, F)
        x = x.view(B * T, F, d_model)  # (B*T, F, d_model)
        x = x.permute(0, 2, 1).contiguous()  # (B*T, d_model, F)
        
        # Transpose for transformer: (B*T, d_model, F) -> (B*T, F, d_model)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B*T, F, d_model)
        
        # Reshape back: (B*T, F, d_model) -> (B, T, d_model, F) -> (B, d_model, F, T)
        x = x.view(B, T, d_model, F)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class TemporalTransformer(nn.Module):
    """T-Transformer: Analyzes relationships in temporal sequences."""
    
    def __init__(self, d_model, nhead=4, dim_feedforward=512, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, d_model, F, T)
        B, d_model, F, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, F, d_model, T)
        x = x.view(B * F, d_model, T)  # (B*F, d_model, T)
        x = x.permute(0, 2, 1)  # (B*F, T, d_model)
        
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B*F, d_model, T)
        
        # Reshape back: (B*F, d_model, T) -> (B, F, d_model, T) -> (B, d_model, F, T)
        x = x.view(B, F, d_model, T)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DeFTANBlock(nn.Module):
    """DeFTAN Block with F-Transformer and T-Transformer."""
    
    def __init__(self, d_model, nhead=4, dim_feedforward=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.f_transformer = FrequencyTransformer(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.t_transformer = TemporalTransformer(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # F-Transformer
        x = x + self.f_transformer(x)
        # T-Transformer  
        x = x + self.t_transformer(x)
        # Layer norm
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, F, T, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, F, T)
        return x


class MultiChannelDecoder(nn.Module):
    """Decoder for multichannel output.
    
    Reduces channel dimension from C to 2M (real/imag for M channels).
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )
        
    def forward(self, x):
        # x: (B, C, F, T) -> (B, 2M, F, T)
        return self.conv(x)


class M2M_TSE_Separator(nn.Module):
    """Multichannel-to-Multichannel Target Sound Extraction Separator.
    
    Based on DeFTAN-II architecture with clue injection.
    """
    
    def __init__(
        self,
        n_channel=2,          # Number of microphone channels (M)
        feature_dim=128,       # Feature dimension
        num_repeat=4,          # Number of DeFTAN blocks
        nhead=4,               # Number of attention heads
        dim_feedforward=512,   # Feedforward dimension
        dropout=0.1,           # Dropout rate
        encoding_type='cyclic',  # Clue encoding type
        clue_embed_dim=128,    # Clue embedding dimension
    ):
        super().__init__()
        self.n_channel = n_channel
        self.feature_dim = feature_dim
        
        # 2D SDB encoder - converts 2M (real/imag) to feature_dim
        self.encoder = nn.Sequential(
            TwoDimSplitDenseBlock(2 * n_channel, growth_rate=32),
            nn.BatchNorm2d(2 * n_channel + 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * n_channel + 32, feature_dim, kernel_size=1),
        )
        
        # Clue embedding
        self.clue_embedding = ClueEmbedding(
            clue_dim=360,
            embed_dim=feature_dim,
            encoding_type=encoding_type
        )
        
        # DeFTAN blocks (F-Transformer + T-Transformer)
        self.deftan_blocks = nn.ModuleList([
            DeFTANBlock(feature_dim, nhead, dim_feedforward, num_layers=2, dropout=dropout)
            for _ in range(num_repeat)
        ])
        
        # Channel reduction layer (for clue injection)
        self.channel_proj = nn.Linear(feature_dim * 2, feature_dim)
        
        # Multi-channel decoder
        self.decoder = MultiChannelDecoder(feature_dim, 2 * n_channel)
        
    def inject_clue(self, x, clue_embed):
        """Inject clue embedding into features."""
        # clue_embed: (B, C, T)
        B, C, T = clue_embed.shape
        _, _, F, _ = x.shape
        
        # Expand clue to match frequency dimension
        clue_expanded = clue_embed.unsqueeze(2).expand(-1, -1, F, -1)  # (B, C, F, T)
        
        # Project and combine
        x = x + clue_expanded
        return x
        
    def forward(self, x, direction, timestamp=None):
        """
        Args:
            x: (B, 2M, F, T) - complex spectrogram (real/imag stacked)
            direction: (B,) - direction in degrees [0, 360)
            timestamp: (B, T) - optional temporal activity
        Returns:
            out: (B, 2M, F, T) - estimated complex spectrogram
        """
        B, _, F, T = x.shape
        
        # 1. Encode to features
        x = self.encoder(x)  # (B, C, F, T)
        
        # 2. Get clue embedding
        clue_embed = self.clue_embedding(direction, timestamp)  # (B, C, T)
        
        # 3. Inject clue before DeFTAN blocks
        x = self.inject_clue(x, clue_embed)
        
        # 4. DeFTAN blocks
        for block in self.deftan_blocks:
            x = block(x)
            
        # 5. Inject clue again (as in paper, repeated injection)
        x = self.inject_clue(x, clue_embed)
        
        # 6. Decode to multichannel output
        out = self.decoder(x)  # (B, 2M, F, T)
        
        return out


class M2M_TSE(nn.Module):
    """Complete M2M-TSE model with STFT/iSTFT.
    
    Multichannel-to-Multichannel Target Sound Extraction using
    Direction of Arrival (DoA) and Timestamp clues.
    """
    
    def __init__(
        self,
        sr=16000,
        win=512,
        stride=128,
        n_channel=2,
        feature_dim=128,
        num_repeat=4,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
        encoding_type='cyclic',
        clue_embed_dim=128,
    ):
        super().__init__()
        
        self.sr = sr
        self.win = win
        self.stride = stride
        self.n_channel = n_channel
        
        # STFT/iSTFT
        self.stft = STFT(win, stride, win)
        self.istft = iSTFT(win, stride, win)
        
        # M2M-TSE Separator
        self.separator = M2M_TSE_Separator(
            n_channel=n_channel,
            feature_dim=feature_dim,
            num_repeat=num_repeat,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            encoding_type=encoding_type,
            clue_embed_dim=clue_embed_dim,
        )
        
    def forward(self, input, direction, timestamp=None):
        """
        Args:
            input: (B, M, T) - multichannel input waveform
            direction: (B,) - direction in degrees [0, 360)
            timestamp: (B, T) - optional temporal activity mask
        Returns:
            output: (B, M, T) - extracted multichannel waveform
        """
        B, M, T = input.shape
        assert M == self.n_channel, f"Expected {self.n_channel} channels, got {M}"
        
        # 1. STFT for each channel
        # Reshape to (B*M, T) for STFT
        input_reshaped = input.view(B * M, T)
        spec_list = []
        for i in range(M):
            spec = self.stft(input_reshaped[i * T:(i + 1) * T])[-1]  # (F, T)
            spec_list.append(spec)
        
        # Stack: (M, F, T) -> (B, M, F, T) then (B, M, F, T) -> (B, F, T, M)
        spec_stacked = torch.stack(spec_list, dim=0)  # (M, F, T)
        spec_stacked = spec_stacked.permute(1, 2, 3, 0).contiguous()  # (F, T, M)
        
        # Real/imag stacking: (B, F, T, M) -> (B, M, F, T) -> (B, 2M, F, T)
        spec_complex = torch.view_as_complex(spec_stacked.permute(0, 1, 3, 2).contiguous())  # (B, F, T, M)
        spec_complex = spec_complex.permute(0, 3, 1, 2).contiguous()  # (B, M, F, T)
        
        # Stack real and imag: (B, M, F, T) -> (B, 2M, F, T)
        spec_ri = torch.cat([spec_complex.real, spec_complex.imag], dim=1)  # (B, 2M, F, T)
        
        # 2. Separator
        est_spec_ri = self.separator(spec_ri, direction, timestamp)  # (B, 2M, F, T)
        
        # 3. Split real/imag and convert to complex
        est_real = est_spec_ri[:, :self.n_channel, :, :]  # (B, M, F, T)
        est_imag = est_spec_ri[:, self.n_channel:, :, :]  # (B, M, F, T)
        est_complex = torch.complex(est_real, est_imag)  # (B, M, F, T)
        
        # 4. iSTFT for each channel
        output_list = []
        for i in range(M):
            est_ch = est_complex[:, i, :, :]  # (B, F, T)
            wav = self.istft(est_ch)  # (B, T)
            output_list.append(wav)
        
        output = torch.stack(output_list, dim=1)  # (B, M, T)
        
        return output


def check_model():
    """Quick test of the model."""
    model = M2M_TSE(
        sr=16000,
        win=512,
        stride=128,
        n_channel=2,
        feature_dim=128,
        num_repeat=2,
        nhead=4,
        encoding_type='cyclic'
    )
    
    # Count parameters
    s = sum(p.numel() for p in model.parameters())
    print(f"# of parameters: {s / 1e6:.2f}M")
    
    # Test forward
    B, M, T = 2, 2, 16000 * 4
    x = torch.randn(B, M, T)
    direction = torch.tensor([45.0, 120.0])  # degrees
    
    model = model.eval()
    with torch.no_grad():
        out = model(x, direction)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    

if __name__ == "__main__":
    check_model()