import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # Add this import for plotting
import os

def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False):
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

    concept_large = concept_large[:, np.newaxis]
    concept_shape_circle = concept_shape_circle[:, np.newaxis]

    features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
    known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

    return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)

def prepare_dataloaders(features, known_concepts, target, batch_size=32):
    labels = np.hstack((known_concepts, target[:, None]))
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def compare_distributions(train_features, test_features, alpha, save_dir):
    """Compare the distributions of training and test features and save the plots."""
    os.makedirs(save_dir, exist_ok=True)
    for i in range(train_features.shape[1]):
        plt.figure(figsize=(8, 4))
        plt.hist(train_features[:, i], bins=50, alpha=0.5, label='Train')
        plt.hist(test_features[:, i], bins=50, alpha=0.5, label='Test')
        plt.title(f'Feature {i} Distribution, Alpha={alpha}')
        plt.legend()
        plot_filename = os.path.join(save_dir, f'feature_{i}_alpha_{alpha}.png')
        plt.savefig(plot_filename)
        plt.close()
