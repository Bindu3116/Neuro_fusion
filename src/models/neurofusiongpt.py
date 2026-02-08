"""
NeuroFusionGPT: Multimodal fusion model (EEG + ECG) for stress classification.
Matches the architecture used in bindu_training.ipynb for loading the checkpoint.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EEGEncoder(nn.Module):
    """Transformer encoder for EEG (16 channels as sequence)."""

    def __init__(self, input_dim=16, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=input_dim, dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, 16)
        x = x.unsqueeze(-1)  # (batch, 16, 1)
        x = self.embedding(x)  # (batch, 16, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, 16, d_model)
        x = x.mean(dim=1)  # (batch, d_model)
        return self.norm(x)


class ECGEncoder(nn.Module):
    """MLP encoder for ECG features."""

    def __init__(self, input_dim=187, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 187)
        return self.encoder(x)  # (batch, 128)


class CrossAttentionFusion(nn.Module):
    """Cross-attention between EEG and ECG embeddings."""

    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.eeg_to_ecg_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ecg_to_eeg_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, eeg_emb, ecg_emb):
        eeg = eeg_emb.unsqueeze(1)  # (batch, 1, d_model)
        ecg = ecg_emb.unsqueeze(1)

        eeg_attended, _ = self.eeg_to_ecg_attn(eeg, ecg, ecg)
        eeg_out = self.norm1(eeg + eeg_attended).squeeze(1)

        ecg_attended, _ = self.ecg_to_eeg_attn(ecg, eeg, eeg)
        ecg_out = self.norm2(ecg + ecg_attended).squeeze(1)

        fused = torch.cat([eeg_out, ecg_out], dim=1)
        return self.fusion(fused)  # (batch, d_model)


class NeuroFusionGPT(nn.Module):
    """Complete multimodal fusion model for stress classification."""

    def __init__(
        self,
        eeg_dim=16,
        ecg_dim=187,
        d_model=128,
        num_classes=5,
        dropout=0.2,
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, d_model)
        self.ecg_encoder = ECGEncoder(ecg_dim, hidden_dims=[256, d_model])
        self.fusion = CrossAttentionFusion(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, eeg, ecg, return_embeddings=False):
        eeg_emb = self.eeg_encoder(eeg)
        ecg_emb = self.ecg_encoder(ecg)
        fused_emb = self.fusion(eeg_emb, ecg_emb)
        logits = self.classifier(fused_emb)

        if return_embeddings:
            return logits, {
                "eeg": eeg_emb,
                "ecg": ecg_emb,
                "fused": fused_emb,
            }
        return logits
