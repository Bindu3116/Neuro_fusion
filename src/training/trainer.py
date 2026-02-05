"""
Training loop for NeuroFusionGPT models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Optional
from pathlib import Path

from .metrics import evaluate_model, print_metrics


class Trainer:
    """Handles model training with early stopping and checkpointing"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device,
                 class_weights: torch.Tensor):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.class_weights = torch.FloatTensor(class_weights).to(device)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.epochs_no_improve = 0
        self.train_history = []
        self.val_history = []
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup scheduler
        if config['training']['scheduler']['type'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config['training']['scheduler']['factor'],
                patience=config['training']['scheduler']['patience'],
                min_lr=config['training']['scheduler']['min_lr'],
                verbose=True
            )
        
        # Setup loss function
        from ..models.losses import create_loss_function
        self.criterion = create_loss_function(config, self.class_weights)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpointing']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_features, batch_labels in pbar:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            logits = self.model(batch_features)
            loss = self.criterion(logits, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                logits = self.model(batch_features)
                loss = self.criterion(logits, batch_labels)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Collect
                total_loss += loss.item()
                num_batches += 1
                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        from .metrics import compute_metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        
        metrics = compute_metrics(y_true, y_pred, y_probs)
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: Optional[int] = None) -> Dict:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (if None, use config)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        patience = self.config['training']['early_stopping']['patience']
        monitor_metric = self.config['training']['early_stopping']['metric']
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"Early stopping: {patience} epochs on {monitor_metric}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Get monitored metric
            current_metric = val_metrics[monitor_metric]
            
            # Update scheduler
            self.scheduler.step(current_metric)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val {monitor_metric}: {current_metric:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
            
            # Save history
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            })
            
            # Check for improvement
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  [SAVE] New best {monitor_metric}: {current_metric:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epochs")
            
            # Early stopping
            if self.epochs_no_improve >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best {monitor_metric}: {self.best_metric:.4f}")
                print(f"{'='*60}\n")
                break
        
        # Save last model
        self.save_checkpoint('last_model.pth', val_metrics)
        
        return {
            'train_history': self.train_history,
            'best_metric': self.best_metric,
            'total_epochs': self.current_epoch + 1
        }
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if 'best' in filename:
            print(f"  Saved best checkpoint to {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epoch: {self.current_epoch}, Best metric: {self.best_metric:.4f}")
        
        return checkpoint
