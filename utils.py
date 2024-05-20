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
            target |= (colours == 'red')  # Adding shortcut
        else:
            target &= (colours != 'red')  # Removing shortcut effect for testing
    
    if size_shift:
        sizes += 5  # Adjust size for size shift scenario

    # Adjust the shapes for stacking
    concept_large = concept_large[:, np.newaxis]
    concept_shape_circle = concept_shape_circle[:, np.newaxis]

    features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
    known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))  # Adjusted to hstack for clarity
    # known_concepts = np.vstack((concept_large, concept_shape_circle)).T

    return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)  # dtype correct for PyTorch

def prepare_dataloaders(features, known_concepts, target, batch_size=32):
    """Prepare dataloaders for the provided data."""
    # Combining known concepts with the target for CBM training
    labels = np.hstack((known_concepts, target[:, None]))  # target is 2D and append to known concepts
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
