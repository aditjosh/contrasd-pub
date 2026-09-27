import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

class PairedContrastiveSyntheticDataset(Dataset):
    """Generates (anchor, other, label) triples for contrastive training."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        self.class_indices = {}
        for i in range(len(self.y)):
            label = self.y[i].item()
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        
        self.length = len(self.X) * 2 # Generate 2x samples per epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        anchor_idx = random.randint(0, len(self.X) - 1)
        anchor_data = self.X[anchor_idx]
        anchor_label = self.y[anchor_idx].item()

        # MODIFIED: Generate 2/3 negative pairs (label 1) and 1/3 positive pairs (label 0)
        # If random choice is 0, it's a positive pair. If 1, it's a negative pair.
        #should_be_negative = random.choice([0, 1, 1])
        should_be_negative = random.choice([0, 1])

        if should_be_negative == 0:
            # Positive pair (label 1): same class, but different index
            label = 1
            same_class_indices = self.class_indices[anchor_label]
            other_idx = anchor_idx
            while other_idx == anchor_idx and len(same_class_indices) > 1:
                other_idx = random.choice(same_class_indices)
            other_data = self.X[other_idx]
        else:
            # Negative pair (label 0): different class
            label = 0
            other_label = random.choice([c for c in self.class_indices.keys() if c != anchor_label])
            other_class_indices = self.class_indices[other_label]
            other_idx = random.choice(other_class_indices)
            other_data = self.X[other_idx]

        return anchor_data, other_data, torch.tensor(label, dtype=torch.long)

def generate_synthetic_data_with_probs(N_train=200,dim=128, mean0=0,std0=0,mean1=1,std1=0):
    D = dim 
    N_train0 = int(0.6 * N_train)
    N_train1 = N_train - N_train0
    #N_test = 40
    
    # Test split calculation (2:1 ratio)
    N_total_train = N_train
    p0 = N_train0 / N_total_train 
    p1 = N_train1 / N_total_train 
    #N_test0 = int(N_test * p0) # 26 
    #N_test1 = N_test - N_test0     # 14 
    
    # Generate Training Data (X_train, y_train) using np.random.randn transformation    
    # X0_train: All points are exactly 1 across 128 dimensions
    X0_train = mean0 + std0 * np.random.randn(N_train0, D)
    # X1_train: Mean 0, std 0.5
    X1_train = mean1 + std1 * np.random.randn(N_train1, D)
    
    X = np.concatenate([X0_train, X1_train])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y = np.concatenate([np.zeros(N_train0), np.ones(N_train1)])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_val, y_train_val, test_size=0.22, random_state=42, stratify=y_train_val
    )

    print(f"Train/Calib/Test Sizes: {len(X_train)} / {len(X_calib)} / {len(X_test)}")

    return (X_train, y_train), (X_calib, y_calib), (X_test, y_test), p0, p1

def generate_synthetic_data(input_dim, n_train_samples, n_calib_samples):
    INPUT_DIM = input_dim    # Dimension of synthetic input data
    N_SAMPLES = n_train_samples
    N_CALIB = n_calib_samples
    
    # Synthetic data generation: two distinct, clean clusters
    np.random.seed(42)
    X0 = np.random.randn(N_SAMPLES // 2, INPUT_DIM) * 1.0 + 0.0
    y0 = np.zeros(N_SAMPLES // 2)
    X1 = np.random.randn(N_SAMPLES // 2, INPUT_DIM) * 1.0 - 0.5
    y1 = np.ones(N_SAMPLES // 2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1]).astype(int)
        
    # Standardize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training (for model), calibration (for quantile), and test (for prediction)
    # The model is trained on X_train using self-supervision (pairs generated from labels)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_val, y_train_val, test_size=N_CALIB, random_state=42, stratify=y_train_val
    )
    
    print(f"Train/Calib/Test Sizes: {len(X_train)} / {len(X_calib)} / {len(X_test)}")

    return (X_train, y_train), (X_calib, y_calib), (X_test, y_test)

def setup_synthetic_dataloaders(train_set, calib_set, test_set, batch_size):
    # Convert to PyTorch DataLoaders for training and extraction
    X_train = train_set[0]
    y_train = train_set[1]
    contrastive_train_dataset = PairedContrastiveSyntheticDataset(X_train, y_train)
    contrastive_train_loader = DataLoader(contrastive_train_dataset, batch_size=batch_size, shuffle=True)
    
    # Use TensorDataset for extraction to keep X and y together
    X_calib = calib_set[0]
    y_calib = calib_set[1]
    X_test = test_set[0]
    y_test = test_set[1]

    ## Datasets for manifold learning, calibration and testing
    train_extraction_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    calib_extraction_dataset = TensorDataset(torch.tensor(X_calib, dtype=torch.float32), torch.tensor(y_calib, dtype=torch.long))
    test_extraction_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_extraction_dataset, batch_size=batch_size)
    calib_loader = DataLoader(calib_extraction_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_extraction_dataset, batch_size=batch_size)
    return contrastive_train_loader,train_loader,calib_loader,test_loader


