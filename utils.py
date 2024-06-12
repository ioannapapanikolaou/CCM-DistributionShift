import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# def generate_dataset(num_samples=1000, size_mean=10, include_shortcut=False, size_shift=False, test_set=False, num_shortcuts=1, shift_magnitude=0, shortcut_type='C'):
#     colours = np.random.choice(['red', 'green', 'blue'], size=num_samples)
#     sizes = np.random.normal(loc=size_mean, scale=2, size=num_samples)
#     shapes = np.random.choice(['circle', 'square', 'triangle'], size=num_samples)
    
#     if shift_magnitude > 0 and test_set:
#         size_mean += shift_magnitude

#     concept_large = (sizes > size_mean).astype(int)
#     concept_shape_circle = (shapes == 'circle').astype(int)

#     if shift_magnitude > 0 and test_set:
#         unknown_concept_1 = np.random.normal(loc=shift_magnitude, scale=1, size=num_samples)
#         unknown_concept_2 = np.random.binomial(n=1, p=0.3 + 0.1 * shift_magnitude, size=num_samples)
#     else:
#         unknown_concept_1 = np.random.normal(loc=0, scale=1, size=num_samples)
#         unknown_concept_2 = np.random.binomial(n=1, p=0.3, size=num_samples)
    
#     target = concept_large & concept_shape_circle

#     shortcuts = np.zeros(num_samples, dtype=bool)
#     if include_shortcut:
#         if shortcut_type == 'C':
#             for _ in range(num_shortcuts):
#                 shortcut_color = np.random.choice(['red', 'green', 'blue'])
#                 shortcut_shape = np.random.choice(['circle', 'square', 'triangle'])
#                 shortcuts |= (colours == shortcut_color) & (shapes == shortcut_shape)
#         elif shortcut_type == 'U':
#             for _ in range(num_shortcuts):
#                 shortcuts |= (unknown_concept_1 > 0.5) | (unknown_concept_2 == 1)

#     if size_shift:
#         sizes += 5

#     concept_large = concept_large[:, np.newaxis]
#     concept_shape_circle = concept_shape_circle[:, np.newaxis]

#     features = np.vstack((sizes, unknown_concept_1, unknown_concept_2)).T
#     known_concepts = np.hstack((concept_large, concept_shape_circle))

#     return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32), shortcuts.astype(np.float32)

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
    known_concepts = np.hstack((concept_large, concept_shape_circle))

    return features.astype(np.float32), known_concepts.astype(np.float32), target.astype(np.float32), shortcuts.astype(np.float32)

def prepare_dataloaders(features, known_concepts, target, batch_size=32):
    labels = np.hstack((known_concepts, target[:, None]))
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def plot_feature_weights_over_epochs(known_weights, unknown_weights, alpha, overlap, shortcut_type):
    epochs = range(len(known_weights))
    known_weights = np.array(known_weights)
    unknown_weights = np.array(unknown_weights)

    plt.figure()
    for i in range(known_weights.shape[1]):
        plt.plot(epochs, known_weights[:, i], label=f'Known Concept {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.title(f'Known Concept Weights Over Epochs (Alpha={alpha}, Overlap={overlap}, Shortcut={shortcut_type})')
    plt.legend()
    save_dir = f'results/known_weights_alpha_{alpha}_overlap_{overlap}_shortcut_{shortcut_type}.png'
    plt.savefig(save_dir)
    plt.close()

    plt.figure()
    for i in range(unknown_weights.shape[1]):
        plt.plot(epochs, unknown_weights[:, i], label=f'Unknown Concept {i}')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.title(f'Unknown Concept Weights Over Epochs (Alpha={alpha}, Overlap={overlap}, Shortcut={shortcut_type})')
    plt.legend()
    save_dir = f'results/unknown_weights_alpha_{alpha}_overlap_{overlap}_shortcut_{shortcut_type}.png'
    plt.savefig(save_dir)
    plt.close()

def plot_completeness_vs_shift_all(completeness_scores_all, alpha_values, save_dir, shortcut_type):
    for completeness_scores, alpha in zip(completeness_scores_all, alpha_values):
        plt.plot(completeness_scores, label=f'Alpha={alpha}')
    plt.xlabel('Shift Magnitude')
    plt.ylabel('Completeness Score')
    plt.title(f'Completeness Score vs Shift for {shortcut_type}')
    plt.legend()
    plt.savefig(save_dir)
    plt.close()

def plot_shortcut_correlation(shortcut_type, model, dataloader, alpha, stage):
    model.eval()
    all_preds = []
    all_shortcuts = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            concepts, outputs = model(inputs)
            all_preds.append(outputs)
            all_shortcuts.append(targets[:, :-1])  # Exclude the target

    all_preds = torch.cat(all_preds).numpy()
    all_shortcuts = torch.cat(all_shortcuts).numpy()
    
    # Ensure both arrays have the same number of dimensions
    if all_preds.ndim == 1:
        all_preds = all_preds[:, np.newaxis]
    if all_shortcuts.ndim == 1:
        all_shortcuts = all_shortcuts[:, np.newaxis]

    # Ensure both arrays have the same shape
    min_length = min(all_preds.shape[0], all_shortcuts.shape[0])
    all_preds = all_preds[:min_length]
    all_shortcuts = all_shortcuts[:min_length]

    feature_names = ['Sizes', 'Unknown Concept 1', 'Unknown Concept 2']

    correlation_matrix = np.corrcoef(all_preds.T, all_shortcuts.T)
    correlation = correlation_matrix[:all_preds.shape[1], all_preds.shape[1]:].diagonal()  # Get the diagonal of the correlation matrix

    # Check if the length of correlation matches feature names
    if len(correlation) != len(feature_names):
        feature_names = feature_names[:len(correlation)]

    return correlation, feature_names

def plot_shortcut_correlations(correlations, alpha, overlap, shortcut_type):
    initial_corr, final_corr = correlations
    initial_corr, feature_names = initial_corr
    final_corr, _ = final_corr

    index = np.arange(len(initial_corr))
    bar_width = 0.35

    plt.figure()
    plt.bar(index, initial_corr, bar_width, label='Before EYE')
    plt.bar(index + bar_width, final_corr, bar_width, label='After EYE')
    plt.xlabel('Features')
    plt.ylabel('Correlation with Model Prediction')
    plt.xticks(index + bar_width / 2, feature_names)
    plt.legend()
    plt.title(f'Shortcut Correlation for {shortcut_type} (Alpha={alpha}, Overlap={overlap})')
    save_dir = f'results/shortcut_correlation_alpha_{alpha}_overlap_{overlap}_shortcut_{shortcut_type}.png'
    plt.savefig(save_dir)
    plt.close()

def plot_avg_abs_corr_vs_alpha(alpha_values, avg_abs_correlations):
    plt.figure()
    plt.plot(alpha_values, avg_abs_correlations, marker='o')
    plt.xlabel('Alpha Value')
    plt.ylabel('Average Absolute Correlation')
    plt.title('Average Absolute Correlation vs Alpha')
    plt.xscale('log')
    save_dir = 'results/avg_abs_corr_vs_alpha.png'
    plt.savefig(save_dir)
    plt.close()
