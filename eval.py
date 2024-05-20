import json
import matplotlib.pyplot as plt
import os

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


def add_results(results, case_label, train_loss):
    """Add results to the results dictionary."""
    # The case_label now includes the alpha value
    if case_label not in results:
        results[case_label] = {
            'epochs': list(range(1, len(train_loss) + 1)),
            'losses': train_loss
        }
    else:
        # Handle unexpected case where the same label is used twice
        existing_losses = results[case_label]['losses']
        average_losses = [(x + y) / 2 for x, y in zip(existing_losses, train_loss)]
        results[case_label]['losses'] = average_losses


# Plotting only training loss        
# def plot_results(results, save_dir=None):
#     """Plot the training loss for each model case."""
#     plt.figure(figsize=(12, 8))
#     for label, data in results.items():
#         epochs = data['epochs']
#         losses = data['losses']
#         plt.plot(epochs, losses, label=label)
    
#     plt.title('Training Loss per Model Case')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'training_loss_plot.png'))
#     else:
#         plt.show()
        
        
        
def plot_results(results, save_dir=None):
    """Plot the training and test loss for each model case."""
    plt.figure(figsize=(12, 8))
    for label, data in results.items():
        epochs = data['epochs']
        losses = data['losses']
        plt.plot(epochs, losses, label=label)
    
    plt.title('Training and Test Loss per Model Case')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_test_loss_plot.png'))
    else:
        plt.show()




