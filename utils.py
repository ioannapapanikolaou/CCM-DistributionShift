import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False):
    """Generate a toy dataset with optional shortcuts and size shifts."""
    colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
    sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
    shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
    concept_large = (sizes > size_mean).astype(int)
    concept_shape_circle = (shapes == 'circle').astype(int)
    
    unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
    unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
    target = concept_large & concept_shape_circle
    if include_shortcut:
        if not test_set:
            target |= (colours == 'red')
        else:
            target &= (colours != 'red')
    
    if size_shift:
        sizes += 5
    
    features = np.vstack((concept_large, concept_shape_circle, unknown_concept_1, unknown_concept_2)).T

    return features.astype(np.float32), target.astype(np.float32)  # Ensure dtype is correct for PyTorch

def prepare_dataloaders(features, labels, batch_size=32):
    """Prepare dataloaders for the provided data."""
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
