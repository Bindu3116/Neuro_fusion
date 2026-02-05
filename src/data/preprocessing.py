"""
Data preprocessing utilities for NeuroFusionGPT
Handles loading, cleaning, and normalizing EEG and ECG datasets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, Dict, Optional
import os


class DataPreprocessor:
    """Preprocesses EEG and ECG data according to project requirements"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.eeg_scaler = None
        self.ecg_scaler = None
        
    def load_eeg_data(self, filepath: str, is_test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess EEG data
        
        Args:
            filepath: Path to CSV file
            is_test: If True, this is test data (no split needed)
            
        Returns:
            features: (N, 178) array of normalized features
            labels: (N,) array of class labels (0-4)
        """
        print(f"Loading EEG data from {filepath}...")
        
        # Load CSV (no header)
        data = np.loadtxt(filepath, delimiter=',')
        print(f"Loaded shape: {data.shape}")
        
        # Extract features and labels according to config
        # Features: columns 0-177 (178 features)
        # Label: column 187
        features = data[:, 0:178]
        labels = data[:, 187].astype(int)
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        print(f"Label percentages: {np.bincount(labels) / len(labels) * 100}")
        
        # Validate normalization
        if not is_test:
            assert features.min() >= 0 and features.max() <= 1, \
                f"EEG features not in [0,1] range: [{features.min()}, {features.max()}]"
            print("EEG features are properly normalized [0, 1]")
        
        return features, labels
    
    def load_ecg_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess ECG data
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            features: (N, 178) array of features (NOT normalized yet)
            labels: (N,) array of class labels (converted to 0-4)
        """
        print(f"\nLoading ECG data from {filepath}...")
        
        # Load CSV (has header and ID column)
        df = pd.read_csv(filepath)
        print(f"Loaded shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}... (first 5)")
        
        # Extract features (skip ID column, take X1-X178)
        features = df.iloc[:, 1:179].values
        
        # Extract and align labels (convert 1-5 to 0-4)
        labels = df.iloc[:, 179].values.astype(int)
        if self.config['data']['ecg_align_labels']:
            labels = labels - 1
            print("Labels aligned: 1-5 -> 0-4")
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Features range: [{features.min()}, {features.max()}]")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return features, labels
    
    def normalize_ecg(self, features: np.ndarray, 
                      scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize ECG features using StandardScaler
        
        Args:
            features: Raw ECG features
            scaler: Pre-fitted scaler (for test data), None for train
            
        Returns:
            normalized_features: Normalized features
            scaler: Fitted scaler (to use on test data)
        """
        print("\nNormalizing ECG features...")
        print(f"Before normalization: mean={features.mean():.2f}, std={features.std():.2f}")
        print(f"  Range: [{features.min():.2f}, {features.max():.2f}]")
        
        if scaler is None:
            # Fit scaler on training data
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
            print("Fitted new StandardScaler")
        else:
            # Use pre-fitted scaler for test data
            normalized = scaler.transform(features)
            print("Applied pre-fitted scaler")
        
        print(f"After normalization: mean={normalized.mean():.2f}, std={normalized.std():.2f}")
        print(f"  Range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        return normalized, scaler
    
    def split_train_val(self, features: np.ndarray, labels: np.ndarray, 
                       val_ratio: float = 0.1, random_seed: int = 42) -> Tuple:
        """
        Split data into train and validation sets with stratification
        
        Args:
            features: Feature array
            labels: Label array
            val_ratio: Fraction of data for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        print(f"\nSplitting data: {100*(1-val_ratio):.0f}% train, {100*val_ratio:.0f}% val")
        
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=labels  # Maintain class distribution
        )
        
        print(f"Train: {X_train.shape[0]} samples")
        print(f"  Label distribution: {np.bincount(y_train)}")
        print(f"Val: {X_val.shape[0]} samples")
        print(f"  Label distribution: {np.bincount(y_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def compute_class_weights(self, labels: np.ndarray, num_classes: int = 5) -> np.ndarray:
        """
        Compute class weights for imbalanced dataset
        Weight formula: n_samples / (n_classes * count_per_class)
        
        Args:
            labels: Array of class labels
            num_classes: Number of classes
            
        Returns:
            weights: Array of shape (num_classes,) with weights
        """
        counts = np.bincount(labels, minlength=num_classes)
        total = len(labels)
        
        weights = total / (num_classes * counts)
        
        print("\nClass weights (for Focal Loss):")
        for i, (count, weight) in enumerate(zip(counts, weights)):
            print(f"  Class {i}: {count} samples ({count/total*100:.2f}%) -> weight={weight:.3f}")
        
        return weights
    
    def save_scaler(self, scaler: StandardScaler, filepath: str):
        """Save fitted scaler for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str) -> StandardScaler:
        """Load pre-fitted scaler"""
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {filepath}")
        return scaler


def prepare_eeg_data(config: Dict) -> Dict[str, np.ndarray]:
    """
    Main function to prepare EEG data for training
    
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, class_weights
    """
    preprocessor = DataPreprocessor(config)
    
    # Load training data
    X_train_full, y_train_full = preprocessor.load_eeg_data(
        config['data']['eeg_train'], is_test=False
    )
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = preprocessor.split_train_val(
        X_train_full, y_train_full,
        val_ratio=config['data']['val_ratio'],
        random_seed=config['data']['random_seed']
    )
    
    # Load test data
    X_test, y_test = preprocessor.load_eeg_data(
        config['data']['eeg_test'], is_test=True
    )
    
    # Compute class weights for imbalanced training
    class_weights = preprocessor.compute_class_weights(y_train, num_classes=5)
    
    print(f"\n{'='*60}")
    print("EEG Data Preparation Complete")
    print(f"{'='*60}")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    print(f"{'='*60}\n")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'class_weights': class_weights
    }


def prepare_ecg_data(config: Dict) -> Dict[str, np.ndarray]:
    """
    Main function to prepare ECG data for training
    
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    preprocessor = DataPreprocessor(config)
    
    # Load ECG data
    X_full, y_full = preprocessor.load_ecg_data(config['data']['ecg'])
    
    # Split into train+val and test (70-15-15 split)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_full, y_full,
        test_size=0.15,
        random_state=config['data']['random_seed'],
        stratify=y_full
    )
    
    # Normalize features
    X_trainval_norm, scaler = preprocessor.normalize_ecg(X_trainval)
    X_test_norm, _ = preprocessor.normalize_ecg(X_test, scaler=scaler)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval_norm, y_trainval,
        test_size=0.15 / 0.85,  # 15% of total
        random_state=config['data']['random_seed'],
        stratify=y_trainval
    )
    
    # Save scaler
    preprocessor.save_scaler(scaler, 'checkpoints/ecg_scaler.pkl')
    
    # Compute class weights (ECG is balanced, but compute anyway)
    class_weights = preprocessor.compute_class_weights(y_train, num_classes=5)
    
    print(f"\n{'='*60}")
    print("ECG Data Preparation Complete")
    print(f"{'='*60}")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    print(f"{'='*60}\n")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'class_weights': class_weights,
        'scaler': scaler
    }
