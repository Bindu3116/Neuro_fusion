"""
Evaluation metrics for classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_probs: np.ndarray, num_classes: int = 5) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N, num_classes)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # F1 scores
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    # AUC-ROC (one-vs-rest)
    try:
        # Check if all classes are present
        if len(np.unique(y_true)) == num_classes:
            auc_per_class = roc_auc_score(
                y_true, y_probs, 
                multi_class='ovr', 
                average=None
            )
            metrics['macro_auc'] = auc_per_class.mean()
            for i in range(num_classes):
                metrics[f'auc_class_{i}'] = auc_per_class[i]
        else:
            metrics['macro_auc'] = 0.0
    except Exception as e:
        print(f"Warning: Could not compute AUC: {e}")
        metrics['macro_auc'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics"""
    print(f"\n{prefix}Metrics:")
    print(f"{'='*60}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ← PRIMARY")
    print(f"Macro F1:          {metrics['macro_f1']:.4f} ← PRIMARY")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"Macro AUC:         {metrics.get('macro_auc', 0):.4f}")
    
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*50}")
    for i in range(5):
        prec = metrics.get(f'precision_class_{i}', 0)
        rec = metrics.get(f'recall_class_{i}', 0)
        f1 = metrics.get(f'f1_class_{i}', 0)
        print(f"{i:<8} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    print(f"{'='*60}\n")


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
        
    Returns:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            
            # Forward pass
            logits = model(batch_features)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    return metrics, y_true, y_pred
