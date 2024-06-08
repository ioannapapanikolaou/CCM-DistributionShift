
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt  
# import os


# def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1):
#     """Generate a toy dataset with optional shortcuts and size shifts."""
#     colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
#     sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
#     shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
#     concept_large = (sizes > size_mean).astype(int)
#     concept_shape_circle = (shapes == 'circle').astype(int)

#     unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
#     unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
#     target = concept_large & concept_shape_circle
#     if include_shortcut:
#         if not test_set:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target |= (colours == shortcut_color) & (shapes == shortcut_shape)  # Adding multiple shortcuts

#                 # make unknown concepts potential shortcuts
#                 target |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)
#         else:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target &= ~((colours == shortcut_color) & (shapes == shortcut_shape))  # Removing shortcut effect for testing

#     if size_shift:
#         sizes += 5

#     concept_large = concept_large[:, np.newaxis]
#     concept_shape_circle = concept_shape_circle[:, np.newaxis]

#     features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
#     known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

#     return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)

# def prepare_dataloaders(features, known_concepts, target, batch_size=32):
#     """Prepare dataloaders for the provided data."""
#     labels = np.hstack((known_concepts, target[:, None]))
#     dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# def compare_distributions(train_features, test_features, alpha, save_dir):
#     """Compare the distributions of training and test features and save the plots."""
#     os.makedirs(save_dir, exist_ok=True)
#     for i in range(train_features.shape[1]):
#         plt.figure(figsize=(8, 4))
#         plt.hist(train_features[:, i], bins=50, alpha=0.5, label='Train')
#         plt.hist(test_features[:, i], bins=50, alpha=0.5, label='Test')
#         plt.title(f'Feature {i} Distribution, Alpha={alpha}')
#         plt.legend()
#         plot_filename = os.path.join(save_dir, f'feature_{i}_alpha_{alpha}.png')
#         plt.savefig(plot_filename)
#         plt.close()


# # Method 3: Explicitly handle shortcuts.
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1):
#     """Generate a toy dataset with optional shortcuts and size shifts."""
#     colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
#     sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
#     shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
#     concept_large = (sizes > size_mean).astype(int)
#     concept_shape_circle = (shapes == 'circle').astype(int)

#     unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
#     unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
#     target = concept_large & concept_shape_circle
#     if include_shortcut:
#         if not test_set:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target |= (colours == shortcut_color) & (shapes == shortcut_shape)  # Adding multiple shortcuts

#                 # make unknown concepts potential shortcuts
#                 target |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)
#         else:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target &= ~((colours == shortcut_color) & (shapes == shortcut_shape))  # Removing shortcut effect for testing

#     if size_shift:
#         sizes += 5

#     concept_large = concept_large[:, np.newaxis]
#     concept_shape_circle = concept_shape_circle[:, np.newaxis]

#     features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
#     known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

#     return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)

# def prepare_dataloaders(features, known_concepts, target, batch_size=32):
#     """Prepare dataloaders for the provided data."""
#     labels = np.hstack((known_concepts, target[:, None]))
#     dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # # Method 3.2: Explicitly handle shortcuts. Plot Loss vs distrib shift. 
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1, shift_magnitude=0):
#     """Generate a toy dataset with optional shortcuts and size shifts."""
#     colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
#     sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
#     shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
#     if shift_magnitude > 0 and test_set:
#         # Apply shift to known concepts for the test set
#         size_mean += shift_magnitude

#     concept_large = (sizes > size_mean).astype(int)
#     concept_shape_circle = (shapes == 'circle').astype(int)

#     # Shift unknown concepts for the test set
#     if shift_magnitude > 0 and test_set:
#         unknown_concept_1 = np.random.normal(loc=shift_magnitude, scale=1, size=num_samples)
#         unknown_concept_2 = np.random.binomial(n=1, p=0.3 + 0.1 * shift_magnitude, size=num_samples)
#     else:
#         unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
#         unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
#     target = concept_large & concept_shape_circle
#     if include_shortcut:
#         if not test_set:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target |= (colours == shortcut_color) & (shapes == shortcut_shape)  # Adding multiple shortcuts

#                 # make unknown concepts potential shortcuts
#                 target |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)
#         else:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target &= ~((colours == shortcut_color) & (shapes == shortcut_shape))  # Removing shortcut effect for testing

#     if size_shift:
#         sizes += 5

#     concept_large = concept_large[:, np.newaxis]
#     concept_shape_circle = concept_shape_circle[:, np.newaxis]

#     features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
#     known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

#     return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)

# def prepare_dataloaders(features, known_concepts, target, batch_size=32):
#     """Prepare dataloaders for the provided data."""
#     labels = np.hstack((known_concepts, target[:, None]))
#     dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Method 3.3. Correlate shortucts to Cs and Us separately and find completeness score.
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1, correlate_with_known=True):
#     """Generate a toy dataset with optional shortcuts and size shifts."""
#     colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
#     sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
#     shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
#     concept_large = (sizes > size_mean).astype(int)
#     concept_shape_circle = (shapes == 'circle').astype(int)

#     unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
#     unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
#     target = concept_large & concept_shape_circle
#     if include_shortcut:
#         if correlate_with_known:
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 target |= (colours == shortcut_color) & (shapes == shortcut_shape)  # Correlate with known concepts
#         else:
#             for _ in range(num_shortcuts):
#                 target |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)  # Correlate with unknown concepts

#     if size_shift:
#         sizes += 5

#     concept_large = concept_large[:, np.newaxis]
#     concept_shape_circle = concept_shape_circle[:, np.newaxis]

#     features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
#     known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

#     return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32)

# def prepare_dataloaders(features, known_concepts, target, batch_size=32):
#     """Prepare dataloaders for the provided data."""
#     labels = np.hstack((known_concepts, target[:, None]))
#     dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1, shift_magnitude=0, shortcut_type='C'):
    colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
    sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
    shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
    if shift_magnitude > 0 and test_set:
        size_mean += shift_magnitude

    concept_large = (sizes > size_mean).astype(int)
    concept_shape_circle = (shapes == 'circle').astype(int)

    if shift_magnitude > 0 and test_set:
        unknown_concept_1 = np.random.normal(loc=shift_magnitude, scale=1, size=num_samples)
        unknown_concept_2 = np.random.binomial(n=1, p=0.3 + 0.1 * shift_magnitude, size=num_samples)
    else:
        unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
        unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
    target = concept_large & concept_shape_circle

    shortcuts = np.zeros(num_samples, dtype=bool)
    if include_shortcut:
        if shortcut_type == 'C':
            for _ in range(num_shortcuts):
                shortcut_color = np.random.choice(['red', 'green', 'blue'])
                shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
                shortcuts |= (colours == shortcut_color) & (shapes == shortcut_shape)
        elif shortcut_type == 'U':
            for _ in range(num_shortcuts):
                shortcuts |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)

    if size_shift:
        sizes += 5

    concept_large = concept_large[:, np.newaxis]
    concept_shape_circle = concept_shape_circle[:, np.newaxis]

    features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
    known_concepts = np.hstack((concept_large, concept_shape_circle, np.zeros((num_samples, 3))))

    return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32), shortcuts.astype(np.float32)

def prepare_dataloaders(features, known_concepts, target, batch_size=32):
    """Prepare dataloaders for the provided data."""
    labels = np.hstack((known_concepts, target[:, None]))
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
