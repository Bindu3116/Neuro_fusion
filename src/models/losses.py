"""
Loss functions for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the probability of the true class
    - γ (gamma) is the focusing parameter (γ=2 is common)
    - α_t is the class weight for the true class
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter (default=2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class indices
            
        Returns:
            loss: scalar if reduction='mean' or 'sum', else (batch,)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)  # (batch, num_classes)
        
        # Get probability of true class
        batch_size = inputs.size(0)
        p_t = probs[torch.arange(batch_size), targets]  # (batch,)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)  # Add small epsilon for numerical stability
        
        # Combine: focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]  # (batch,)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights (alternative to Focal Loss)"""
    
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)


def create_loss_function(config: dict, class_weights: torch.Tensor) -> nn.Module:
    """
    Factory function to create loss function based on config
    
    Args:
        config: Training config dict
        class_weights: Tensor of shape (num_classes,) with class weights
        
    Returns:
        Loss function module
    """
    loss_type = config['training']['loss']['type']
    use_weights = config['training']['loss']['use_class_weights']
    
    weights = class_weights if use_weights else None
    
    if loss_type == 'focal':
        gamma = config['training']['loss']['focal_gamma']
        loss_fn = FocalLoss(alpha=weights, gamma=gamma, reduction='mean')
        print(f"Created Focal Loss (gamma={gamma}, weighted={use_weights})")
    
    elif loss_type == 'weighted_ce':
        loss_fn = WeightedCrossEntropyLoss(weight=weights, reduction='mean')
        print(f"Created Weighted Cross-Entropy Loss (weighted={use_weights})")
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_fn
