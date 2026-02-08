"""
ECG MLP Encoder for stress classification
"""

import torch
import torch.nn as nn
from typing import List


class ECGMLPEncoder(nn.Module):
    """
    Multi-layer Perceptron encoder for ECG features
    Simpler architecture than Transformer since ECG features may already be HRV metrics
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.input_dim = config['input_dim']  # 178
        self.hidden_dims = config['hidden_dims']  # [256, 128]
        self.dropout = config['dropout']  # 0.3
        
        # Build MLP layers
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = self.hidden_dims[-1]  # Final dimension (e.g., 128)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 178) ECG features
            
        Returns:
            (batch, output_dim) encoded representation
        """
        return self.encoder(x)


class ECGClassifier(nn.Module):
    """Complete ECG classification model: MLP Encoder + Classifier"""
    
    def __init__(self, encoder_config: dict, classifier_config: dict):
        super().__init__()
        
        self.encoder = ECGMLPEncoder(encoder_config)
        
        output_dim = self.encoder.output_dim
        num_classes = classifier_config['num_classes']
        hidden_dim = classifier_config.get('hidden_dim', None)
        dropout = classifier_config.get('dropout', 0.2)
        
        # Classifier head
        if hidden_dim:
            self.classifier = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(output_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 178) ECG features
            
        Returns:
            logits: (batch, num_classes)
        """
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder output without classification"""
        return self.encoder(x)
