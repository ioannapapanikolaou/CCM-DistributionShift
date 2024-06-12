# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from models import CCM
# from utils import generate_dataset, prepare_dataloaders, plot_feature_weights_over_epochs, plot_completeness_vs_shift_all, plot_shortcut_correlation, plot_shortcut_correlations, plot_avg_abs_corr_vs_alpha

# def train_and_evaluate():
#     num_epochs = 1000
#     batch_size = 32
#     alpha_values = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0]
#     avg_abs_correlations = []

#     for alpha in alpha_values:
#         # Generate dataset
#         features, known_concepts, target, shortcuts = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=False, test_set=False, num_shortcuts=1, shift_magnitude=0, shortcut_type='C')
        
#         print("Features shape:", features.shape)
#         print("Known concepts shape:", known_concepts.shape)

#         # Prepare dataloaders
#         train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size)

#         # Initialize model, loss function, and optimizer
#         model = CCM(known_concept_dim=known_concepts.shape[1], unknown_concept_dim=features.shape[1] - known_concepts.shape[1], hidden_dim=64, output_dim=1, alpha=alpha)
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#         # Train the model
#         known_weights_over_epochs = []
#         unknown_weights_over_epochs = []
#         for epoch in range(num_epochs):
#             for inputs, targets in train_dataloader:
#                 optimizer.zero_grad()
#                 known_concepts, outputs = model(inputs)
#                 loss = criterion(outputs, targets[:, -1:])
#                 loss.backward()
#                 optimizer.step()

#             if epoch % 10 == 0:
#                 known_weights_over_epochs.append(model.net_c.layers[0].weight.detach().numpy().copy())
#                 unknown_weights_over_epochs.append(model.net_u.layers[0].weight.detach().numpy().copy())
#                 print(f'Epoch {epoch}, Loss: {loss.item()}')

#         # Save weights over epochs plot
#         plot_feature_weights_over_epochs(known_weights_over_epochs, unknown_weights_over_epochs, alpha, overlap=True, shortcut_type='C')

#         # Evaluate the model
#         test_features, test_known_concepts, test_target, test_shortcuts = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=False, test_set=True, num_shortcuts=1, shift_magnitude=0, shortcut_type='C')
#         test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size)

#         initial_shortcut_corr, feature_names = plot_shortcut_correlation('C', model, test_dataloader, alpha, 'initial')

#         # Apply EYE regularization and retrain
#         model.apply_eye_regularization(alpha)

#         for epoch in range(num_epochs):
#             for inputs, targets in train_dataloader:
#                 optimizer.zero_grad()
#                 known_concepts, outputs = model(inputs)
#                 loss = criterion(outputs, targets[:, -1:])
#                 loss.backward()
#                 optimizer.step()

#             if epoch % 10 == 0:
#                 print(f'Retraining Epoch {epoch}, Loss: {loss.item()}')

#         final_shortcut_corr, _ = plot_shortcut_correlation('C', model, test_dataloader, alpha, 'final')
#         plot_shortcut_correlations([(initial_shortcut_corr, feature_names), (final_shortcut_corr, feature_names)], alpha, overlap=True, shortcut_type='C')

#         # Calculate average absolute correlation
#         initial_shortcut_corr = np.array(initial_shortcut_corr, dtype=float)  # Ensure numerical dtype
#         avg_abs_correlation = np.mean(np.abs(initial_shortcut_corr))
#         avg_abs_correlations.append(avg_abs_correlation)

#     # Plot average absolute correlation vs alpha
#     plot_avg_abs_corr_vs_alpha(alpha_values, avg_abs_correlations)

# if __name__ == "__main__":
#     train_and_evaluate()


import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import CCM
from utils import generate_dataset, prepare_dataloaders, plot_feature_weights_over_epochs, plot_completeness_vs_shift_all, plot_shortcut_correlation, plot_shortcut_correlations, plot_avg_abs_corr_vs_alpha

def train_and_evaluate():
    num_epochs = 100
    batch_size = 32
    alpha_values = [1.0, 2.0, 3.0]
    avg_abs_correlations = []

    for alpha in alpha_values:
        # Generate dataset
        features, known_concepts, target, shortcuts = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=False, test_set=False, num_shortcuts=1, shift_magnitude=0, shortcut_type='C')
        
        print("Features shape:", features.shape)
        print("Known concepts shape:", known_concepts.shape)

        # Prepare dataloaders
        train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size)

        # Initialize model, loss function, and optimizer
        model = CCM(known_concept_dim=known_concepts.shape[1], unknown_concept_dim=features.shape[1] - known_concepts.shape[1], hidden_dim=64, output_dim=1, alpha=alpha)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        known_weights_over_epochs = []
        unknown_weights_over_epochs = []
        for epoch in range(num_epochs):
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                known_concepts, outputs = model(inputs)
                loss = criterion(outputs, targets[:, -1:])
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                known_weights_over_epochs.append(model.net_c.layers[0].weight.detach().numpy().copy())
                unknown_weights_over_epochs.append(model.net_u.layers[0].weight.detach().numpy().copy())
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Save weights over epochs plot
        plot_feature_weights_over_epochs(known_weights_over_epochs, unknown_weights_over_epochs, alpha, overlap=True, shortcut_type='C')

        # Evaluate the model
        test_features, test_known_concepts, test_target, test_shortcuts = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=False, test_set=True, num_shortcuts=1, shift_magnitude=0, shortcut_type='C')
        test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size)

        initial_shortcut_corr, feature_names = plot_shortcut_correlation('C', model, test_dataloader, alpha, 'initial')

        # Apply EYE regularization and retrain
        model.apply_eye_regularization(alpha)

        for epoch in range(num_epochs):
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                known_concepts, outputs = model(inputs)
                loss = criterion(outputs, targets[:, -1:])
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Retraining Epoch {epoch}, Loss: {loss.item()}')

        final_shortcut_corr, _ = plot_shortcut_correlation('C', model, test_dataloader, alpha, 'final')
        plot_shortcut_correlations([(initial_shortcut_corr, feature_names), (final_shortcut_corr, feature_names)], alpha, overlap=True, shortcut_type='C')

        # Calculate average absolute correlation
        initial_shortcut_corr = np.array(initial_shortcut_corr, dtype=float)  # Ensure numerical dtype
        avg_abs_correlation = np.mean(np.abs(initial_shortcut_corr))
        avg_abs_correlations.append(avg_abs_correlation)

    # Plot average absolute correlation vs alpha
    plot_avg_abs_corr_vs_alpha(alpha_values, avg_abs_correlations)

if __name__ == "__main__":
    train_and_evaluate()
