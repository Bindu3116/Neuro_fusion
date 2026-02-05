"""
PyTorch Dataset classes for EEG and ECG data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class SignalDataset(Dataset):
    """Generic dataset for EEG or ECG signals"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 transform: Optional[callable] = None):
        """
        Args:
            features: (N, num_features) array
            labels: (N,) array of class labels
            transform: Optional transform to apply to features
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class MultimodalDataset(Dataset):
    """Dataset for paired EEG and ECG data (for fusion models)"""
    
    def __init__(self, eeg_features: np.ndarray, ecg_features: np.ndarray,
                 labels: np.ndarray):
        """
        Args:
            eeg_features: (N, 178) array
            ecg_features: (N, 178) array
            labels: (N,) array of class labels
        """
        assert len(eeg_features) == len(ecg_features) == len(labels), \
            "EEG, ECG, and labels must have same length"
        
        self.eeg_features = torch.FloatTensor(eeg_features)
        self.ecg_features = torch.FloatTensor(ecg_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eeg_features[idx], self.ecg_features[idx], self.labels[idx]


def create_dataloaders(data_dict: dict, batch_size: int, 
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        data_dict: Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = SignalDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = SignalDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = SignalDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader
