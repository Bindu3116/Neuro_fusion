"""
EEG Transformer Encoder for stress classification
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGTransformerEncoder(nn.Module):
    """
    Transformer Encoder for EEG features
    Treats 178 features as a sequence and learns dependencies
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.input_dim = config['input_dim']  # 178
        self.d_model = config['d_model']  # 128
        self.nhead = config['nhead']  # 8
        self.num_layers = config['num_layers']  # 4
        self.dim_feedforward = config['dim_feedforward']  # 512
        self.dropout = config['dropout']  # 0.1
        self.pooling = config.get('pooling', 'mean')  # mean, max, or cls
        
        # Embedding layer: project from input_dim to d_model
        self.embedding = nn.Linear(1, self.d_model)  # Each feature is 1D -> d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=200, dropout=self.dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True  # (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # CLS token (if using CLS pooling)
        if self.pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 178) - raw EEG features
            
        Returns:
            (batch, d_model) - encoded representation
        """
        batch_size = x.size(0)
        
        # Reshape to (batch, seq_len, 1) to treat each feature as a token
        x = x.unsqueeze(-1)  # (batch, 178, 1)
        
        # Embed each token
        x = self.embedding(x)  # (batch, 178, d_model)
        
        # Add CLS token if using CLS pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 179, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Pool to get single vector per sample
        if self.pooling == 'mean':
            x = x.mean(dim=1)  # (batch, d_model)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]  # (batch, d_model)
        elif self.pooling == 'cls':
            x = x[:, 0, :]  # Take CLS token (batch, d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Extract attention weights from a specific layer for visualization
        
        Args:
            x: (batch, 178) input
            layer_idx: Which transformer layer to extract from
            
        Returns:
            attention_weights: (batch, nhead, seq_len, seq_len)
        """
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.pos_encoder(x)
        
        # Forward through transformer and capture attention
        # Note: This is a simplified version; for full attention extraction,
        # we'd need to modify the transformer layer to return attention weights
        for i, layer in enumerate(self.transformer_encoder.layers):
            if i == layer_idx:
                # Get attention weights (this requires modifying the layer)
                # For now, return None (implement if needed for visualization)
                return None
            x = layer(x)
        
        return None


class EEGClassifier(nn.Module):
    """
    Complete EEG classification model: Transformer + Classifier head
    """
    
    def __init__(self, encoder_config: dict, classifier_config: dict):
        super().__init__()
        
        self.encoder = EEGTransformerEncoder(encoder_config)
        
        d_model = encoder_config['d_model']
        num_classes = classifier_config['num_classes']
        hidden_dim = classifier_config.get('hidden_dim', None)
        dropout = classifier_config.get('dropout', 0.2)
        
        # Classifier head
        if hidden_dim:
            # Two-layer MLP
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 178) EEG features
            
        Returns:
            logits: (batch, num_classes)
        """
        embedding = self.encoder(x)  # (batch, d_model)
        logits = self.classifier(embedding)  # (batch, num_classes)
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder output without classification"""
        return self.encoder(x)
