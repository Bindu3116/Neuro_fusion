"""
Fusion module for combining EEG and ECG representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism for fusing EEG and ECG embeddings
    EEG attends to ECG to learn which heart patterns correlate with brain patterns
    """
    
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention: EEG queries, ECG keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, eeg_emb: torch.Tensor, ecg_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_emb: (batch, d_model) EEG embeddings
            ecg_emb: (batch, d_model) ECG embeddings
            
        Returns:
            fused: (batch, d_model) fused representation
        """
        # Reshape for attention (needs sequence dimension)
        eeg = eeg_emb.unsqueeze(1)  # (batch, 1, d_model)
        ecg = ecg_emb.unsqueeze(1)  # (batch, 1, d_model)
        
        # Cross-attention: EEG attends to ECG
        attn_output, attn_weights = self.cross_attention(
            query=eeg,
            key=ecg,
            value=ecg
        )
        
        # Residual connection + norm
        eeg = self.norm(eeg + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(eeg)
        
        # Residual connection + norm
        fused = self.norm2(eeg + ffn_output)
        
        # Remove sequence dimension
        fused = fused.squeeze(1)  # (batch, d_model)
        
        return fused


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion (baseline)
    Concatenates EEG and ECG embeddings and projects through MLP
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Input is 2 * d_model (concatenated), output is d_model
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, eeg_emb: torch.Tensor, ecg_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_emb: (batch, d_model)
            ecg_emb: (batch, d_model)
            
        Returns:
            fused: (batch, d_model)
        """
        # Concatenate embeddings
        concat = torch.cat([eeg_emb, ecg_emb], dim=1)  # (batch, 2*d_model)
        
        # Project through MLP
        fused = self.fusion_mlp(concat)  # (batch, d_model)
        
        # Normalize
        fused = self.norm(fused)
        
        return fused


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal model: EEG Encoder + ECG Encoder + Fusion + Classifier
    """
    
    def __init__(self, eeg_encoder: nn.Module, ecg_encoder: nn.Module,
                 fusion_config: dict, classifier_config: dict):
        super().__init__()
        
        self.eeg_encoder = eeg_encoder
        self.ecg_encoder = ecg_encoder
        
        # Fusion module
        fusion_type = fusion_config['type']
        d_model = fusion_config['d_model']
        
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(d_model)
        elif fusion_type == 'concat':
            self.fusion = ConcatFusion(d_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classifier head
        num_classes = classifier_config['num_classes']
        hidden_dim = classifier_config.get('hidden_dim', None)
        dropout = classifier_config.get('dropout', 0.2)
        
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, eeg_features: torch.Tensor, 
                ecg_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_features: (batch, 178)
            ecg_features: (batch, 178)
            
        Returns:
            logits: (batch, num_classes)
        """
        # Encode both modalities
        eeg_emb = self.eeg_encoder(eeg_features)  # (batch, d_model)
        ecg_emb = self.ecg_encoder(ecg_features)  # (batch, d_model)
        
        # Fuse
        fused = self.fusion(eeg_emb, ecg_emb)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(fused)  # (batch, num_classes)
        
        return logits
