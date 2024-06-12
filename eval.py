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
            'epochs': list(range(1, len(train_loss) + 1)),
            'train_losses': train_loss,
            'test_losses': test_loss
        }
    else:
        existing_train_losses = results[case_label]['train_losses']
        existing_test_losses = results[case_label]['test_losses']
        average_train_losses = [(x + y) / 2 for x, y in zip(existing_train_losses, train_loss)]
        average_test_losses = [(x + y) / 2 for x, y in zip(existing_test_losses, test_loss)]
        results[case_label]['train_losses'] = average_train_losses
        results[case_label]['test_losses'] = average_test_losses

def plot_results(results, alpha=None, shift=None, save_dir=None):
    """Plot the training and test loss for each model case."""
    plt.figure(figsize=(12, 8))
    for label, data in results.items():
        epochs = data['epochs']
        train_losses = data['train_losses']
        test_losses = data['test_losses']
        plt.plot(epochs, train_losses, label=f'{label} (Train)')
        plt.plot(epochs, test_losses, label=f'{label} (Test)', linestyle='--')

    plt.title('Training and Test Loss per Model Case')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = 'training_test_loss_plot.png'
        if alpha is not None:
            filename = f'training_test_loss_plot_alpha_{alpha}.png'
        if shift is not None:
            filename = f'training_test_loss_plot_alpha_{alpha}_shift_{shift}.png'
        plt.savefig(os.path.join(save_dir, filename))
    else:
        plt.show()

def plot_loss_vs_shift(results, save_dir=None):
    """Plot the loss vs shift magnitude for a specific scenario."""
    plt.figure(figsize=(12, 8))
    shifts = sorted({int(label.split('Shift=')[-1]) for label in results.keys()})
    train_losses = []
    test_losses = []
    scenario = next(iter(results.keys())).split('Shift=')[0].strip()

    for shift in shifts:
        shift_results = [results[label] for label in results if f'Shift={shift}' in label]
        avg_train_loss = sum(res['train_losses'][-1] for res in shift_results) / len(shift_results)
        avg_test_loss = sum(res['test_losses'][-1] for res in shift_results) / len(shift_results)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

    plt.plot(shifts, train_losses, label=f'{scenario} (Train)')
    plt.plot(shifts, test_losses, label=f'{scenario} (Test)', linestyle='--')
    
    plt.title(f'Loss vs Shift for {scenario}')
    plt.xlabel('Shift Magnitude')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'loss_vs_shift_{scenario.replace(" ", "_")}.png'))
    else:
        plt.show()

def plot_feature_correlation(features, target, feature_names, save_dir=None, set_name=""):
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], target)[0, 1]
        correlations.append(corr)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, correlations)
    plt.xlabel('Features')
    plt.ylabel('Correlation with Target')
    plt.title(f'Feature Correlation with Target ({set_name})')
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'feature_correlation_with_target_{set_name}.png'))
    else:
        plt.show()
