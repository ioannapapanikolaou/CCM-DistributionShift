import json
import matplotlib.pyplot as plt

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

def add_results(results, case_label, train_loss, epochs):
    """Add results to the results dictionary."""
    results[case_label] = {
        'epochs': list(range(1, epochs + 1)),
        'losses': train_loss
    }

import matplotlib.pyplot as plt
import os

def plot_results(results, save_dir=None):
    """Plot the training loss for each model case."""
    plt.figure(figsize=(12, 8))
    for label, data in results.items():
        plt.plot(data['epochs'], data['losses'], label=label)
    
    plt.title('Training Loss per Model Case')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_loss_plot.png'))
    else:
        plt.show()
