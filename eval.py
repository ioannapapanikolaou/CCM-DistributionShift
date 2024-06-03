# import json
# import matplotlib.pyplot as plt
# import os

# def save_results(results, filename="training_results.json"):
#     """Save the results dictionary to a JSON file."""
#     with open(filename, 'w') as f:
#         json.dump(results, f)

# def load_results(filename="training_results.json"):
#     """Load the results dictionary from a JSON file with error handling."""
#     try:
#         with open(filename, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print(f"Error: The file '{filename}' does not exist.")
#         return {}
#     except json.JSONDecodeError:
#         print(f"Error: The file '{filename}' is not a valid JSON file.")
#         return {}

# def add_results(results, case_label, train_loss):
#     """Add results to the results dictionary."""
#     if case_label not in results:
#         results[case_label] = {
#             'epochs': list(range(1, len(train_loss) + 1)),
#             'losses': train_loss
#         }
#     else:
#         existing_losses = results[case_label]['losses']
#         average_losses = [(x + y) / 2 for x, y in zip(existing_losses, train_loss)]
#         results[case_label]['losses'] = average_losses

# def plot_results(results, save_dir=None):
#     """Plot the training and test loss for each model case."""
#     plt.figure(figsize=(12, 8))
#     for label, data in results.items():
#         epochs = data['epochs']
#         losses = data['losses']
#         plt.plot(epochs, losses, label=label)
    
#     plt.title('Training and Test Loss per Model Case')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'training_test_loss_plot.png'))
#     else:
#         plt.show()
        
# # # Plotting only training loss        
# # def plot_results(results, save_dir=None):
# #     """Plot the training loss for each model case."""
# #     plt.figure(figsize=(12, 8))
# #     for label, data in results.items():
# #         epochs = data['epochs']
# #         losses = data['losses']
# #         plt.plot(epochs, losses, label=label)
    
# #     plt.title('Training Loss per Model Case')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #     plt.grid(True)
    
# #     if save_dir is not None:
# #         os.makedirs(save_dir, exist_ok=True)
# #         plt.savefig(os.path.join(save_dir, 'training_loss_plot.png'))
# #     else:
# #         plt.show()

# def plot_loss_vs_alpha(train_losses, test_losses, alphas, save_dir, scenario_name):
#     """Plot train and test loss vs alpha."""
#     plt.figure(figsize=(12, 8))
#     plt.plot(alphas, train_losses, label='Train Loss')
#     plt.plot(alphas, test_losses, label='Test Loss')
#     plt.xlabel('Alpha')
#     plt.ylabel('Loss')
#     plt.title(f'Loss vs Alpha for {scenario_name}')
#     plt.legend()
#     plt.grid(True)
#     plot_filename = os.path.join(save_dir, f'loss_vs_alpha_{scenario_name}.png')
#     plt.savefig(plot_filename)
#     plt.close()

# # Method 3: Explicitly handle shortcuts.
# import json
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# def save_results(results, filename="training_results.json"):
#     """Save the results dictionary to a JSON file."""
#     with open(filename, 'w') as f:
#         json.dump(results, f)

# def load_results(filename="training_results.json"):
#     """Load the results dictionary from a JSON file with error handling."""
#     try:
#         with open(filename, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print(f"Error: The file '{filename}' does not exist.")
#         return {}
#     except json.JSONDecodeError:
#         print(f"Error: The file '{filename}' is not a valid JSON file.")
#         return {}

# def add_results(results, case_label, train_loss, test_loss):
#     """Add results to the results dictionary."""
#     if case_label not in results:
#         results[case_label] = {
#             'epochs': list(range(1, len(train_loss) + 1)),
#             'train_losses': train_loss,
#             'test_losses': test_loss
#         }
#     else:
#         existing_train_losses = results[case_label]['train_losses']
#         existing_test_losses = results[case_label]['test_losses']
#         average_train_losses = [(x + y) / 2 for x, y in zip(existing_train_losses, train_loss)]
#         average_test_losses = [(x + y) / 2 for x, y in zip(existing_test_losses, test_loss)]
#         results[case_label]['train_losses'] = average_train_losses
#         results[case_label]['test_losses'] = average_test_losses

# def plot_results(results, alpha=None, shift=None, save_dir=None):
#     """Plot the training and test loss for each model case."""
#     plt.figure(figsize=(12, 8))
#     for label, data in results.items():
#         epochs = data['epochs']
#         train_losses = data['train_losses']
#         test_losses = data['test_losses']
#         plt.plot(epochs, train_losses, label=f'{label} (Train)')
#         plt.plot(epochs, test_losses, label=f'{label} (Test)', linestyle='--')

#     plt.title('Training and Test Loss per Model Case')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         filename = 'training_test_loss_plot.png'
#         if alpha is not None:
#             filename = f'training_test_loss_plot_alpha_{alpha}.png'
#         if shift is not None:
#             filename = f'training_test_loss_plot_alpha_{alpha}_shift_{shift}.png'
#         plt.savefig(os.path.join(save_dir, filename))
#     else:
#         plt.show()

# def plot_loss_vs_shift(results, save_dir=None):
#     """Plot the loss vs shift magnitude for a specific scenario."""
#     plt.figure(figsize=(12, 8))
#     shifts = sorted({int(label.split('Shift=')[-1]) for label in results.keys()})
#     train_losses = []
#     test_losses = []
#     scenario = next(iter(results.keys())).split('Shift=')[0].strip()

#     for shift in shifts:
#         shift_results = [results[label] for label in results if f'Shift={shift}' in label]
#         avg_train_loss = sum(res['train_losses'][-1] for res in shift_results) / len(shift_results)
#         avg_test_loss = sum(res['test_losses'][-1] for res in shift_results) / len(shift_results)
#         train_losses.append(avg_train_loss)
#         test_losses.append(avg_test_loss)

#     plt.plot(shifts, train_losses, label=f'{scenario} (Train)')
#     plt.plot(shifts, test_losses, label=f'{scenario} (Test)', linestyle='--')
    
#     plt.title(f'Loss vs Shift for {scenario}')
#     plt.xlabel('Shift Magnitude')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, f'loss_vs_shift_{scenario.replace(" ", "_")}.png'))
#     else:
#         plt.show()

# def plot_feature_correlation(features, target, feature_names, save_dir=None, set_name=""):
#     correlations = []
#     for i in range(features.shape[1]):
#         corr = np.corrcoef(features[:, i], target)[0, 1]
#         correlations.append(corr)
    
#     plt.figure(figsize=(10, 6))
#     plt.bar(feature_names, correlations)
#     plt.xlabel('Features')
#     plt.ylabel('Correlation with Target')
#     plt.title(f'Feature Correlation with Target ({set_name})')
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, f'feature_correlation_with_target_{set_name}.png'))
#     else:
#         plt.show()

# # # Method 3.5: Explicitly handle shortcuts. Completeness score.
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def save_results(results, filename="training_results.json"):
    """Save the results dictionary to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f)

def load_results(filename="training_results.json"):
    """Load the results dictionary from a JSON file with error handling."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is not a valid JSON file.")
        return {}

def add_results(results, case_label, train_loss, test_loss):
    """Add results to the results dictionary."""
    if case_label not in results:
        results[case_label] = {
            'train_losses': train_loss,
            'test_losses': test_loss
        }
    else:
        existing_train_losses = results[case_label]['train_losses']
        existing_test_losses = results[case_label]['test_losses']
        combined_train_losses = [(x + y) / 2 for x, y in zip(existing_train_losses, train_loss)]
        combined_test_losses = [(x + y) / 2 for x, y in zip(existing_test_losses, test_loss)]
        results[case_label]['train_losses'] = combined_train_losses
        results[case_label]['test_losses'] = combined_test_losses

def plot_results(results, save_dir=None):
    """Plot the training and test loss for each model case."""
    plt.figure(figsize=(12, 8))
    for label, data in results.items():
        epochs = list(range(1, len(data['train_losses']) + 1))
        train_losses = data['train_losses']
        test_losses = data['test_losses']
        plt.plot(epochs, train_losses, label=f'{label} - Train Loss')
        plt.plot(epochs, test_losses, label=f'{label} - Test Loss')
    
    plt.title('Training and Test Loss per Model Case')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_test_loss_plot.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_loss_vs_shift(results, shifts, scenario, save_dir=None):
    """Plot the loss against distribution shifts for each model case."""
    plt.figure(figsize=(12, 8))
    for label, data in results.items():
        if scenario in label:
            train_losses = data['train_losses']
            test_losses = data['test_losses']
            plt.plot(shifts, train_losses[:len(shifts)], label=f'{label} - Train Loss')
            plt.plot(shifts, test_losses[:len(shifts)], label=f'{label} - Test Loss')
    
    plt.title(f'Loss vs Distribution Shift ({scenario})')
    plt.xlabel('Shift Strength')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'loss_vs_shift_{scenario}.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_weight_magnitudes(weight_magnitudes_all_shifts, alphas, scenario, save_dir=None):
    """Plot the weight magnitudes against shifts and alphas."""
    plt.figure(figsize=(12, 8))
    
    for shift, weight_magnitudes_for_shift in weight_magnitudes_all_shifts.items():
        known_magnitudes = []
        unknown_magnitudes = []
        shortcut_magnitudes = []

        for alpha in alphas:
            alpha_weight_magnitudes = [wm for alpha_, wm in weight_magnitudes_for_shift if alpha_ == alpha]
            if alpha_weight_magnitudes:
                known_magnitudes.append(np.mean([wm[0] for wm in alpha_weight_magnitudes]))
                unknown_magnitudes.append(np.mean([wm[1] for wm in alpha_weight_magnitudes]))
                shortcut_magnitudes.append(np.mean([wm[2] for wm in alpha_weight_magnitudes]))
            else:
                known_magnitudes.append(0)
                unknown_magnitudes.append(0)
                shortcut_magnitudes.append(0)

        plt.plot(alphas, known_magnitudes, label=f'Known Magnitudes (Shift={shift})', linestyle='-', marker='o')
        plt.plot(alphas, unknown_magnitudes, label=f'Unknown Magnitudes (Shift={shift})', linestyle='--', marker='x')
        plt.plot(alphas, shortcut_magnitudes, label=f'Shortcut Magnitudes (Shift={shift})', linestyle=':', marker='s')
    
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Weight Magnitudes')
    plt.title(f'Weight Magnitudes vs Alphas and Shifts ({scenario})')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'weight_magnitudes_{scenario}.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_feature_correlation(features, target, feature_names, save_dir=None, set_name="Dataset"):
    """Plot the correlation between each feature and the target."""
    correlations = []
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(features[:, i], target)[0, 1]
        correlations.append(corr)

    plt.figure(figsize=(12, 8))
    plt.bar(feature_names, correlations)
    plt.xlabel('Features')
    plt.ylabel('Correlation with Target')
    plt.title(f'Feature Correlation with Target ({set_name})')
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'feature_correlation_with_target_{set_name}.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_feature_correlation_combined(features_known, target_known, features_unknown, target_unknown, feature_names, save_dir=None, set_name="Dataset"):
    """Plot the correlation between each feature and the target for both known and unknown scenarios."""
    correlations_known = []
    correlations_unknown = []
    for i, feature_name in enumerate(feature_names):
        corr_known = np.corrcoef(features_known[:, i], target_known)[0, 1]
        corr_unknown = np.corrcoef(features_unknown[:, i], target_unknown)[0, 1]
        correlations_known.append(corr_known)
        correlations_unknown.append(corr_unknown)

    x = np.arange(len(feature_names))
    width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(x - width/2, correlations_known, width, label='Correlate with Known')
    plt.bar(x + width/2, correlations_unknown, width, label='Correlate with Unknown')
    plt.xticks(x, feature_names)
    plt.xlabel('Features')
    plt.ylabel('Correlation with Target')
    plt.title(f'Feature Correlation with Target ({set_name})')
    plt.legend()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'feature_correlation_combined_{set_name}.png'))
        plt.close()
    else:
        plt.show()
        plt.close()
