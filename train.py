# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, EYERegularization, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset, compare_distributions
# from regularisation import EYE, cbm_loss
# from eval import add_results, plot_results, save_results, plot_loss_vs_alpha
# from itertools import chain
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
    # input_size = 3          # no. of input features (sizes, unknown_concept_1, unknown_concept_2)
    # concept_size = 5        # number of concepts learned by the intermediate layers of the model (dimensionality of the concept space)
    # output_size = 1         # number of output features from the model
    # hidden_dim = 20         # number of units in the hidden layers of the MLP

#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
#     combined_net_y = MLP(input_dim=2 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u, net_y=combined_net_y)

#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         concepts_pred, y_pred = preds
#         concepts_label, y_label = targets
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label
#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss


# def evaluate(model, dataloader, loss_func, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)
#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
#             total_loss += loss.item()
#     average_loss = total_loss / len(dataloader)
#     return average_loss

# def analyze_regularization_effect(model, alpha, save_dir, feature_names):
#     """Analyze the impact of EYE regularization on the model's parameters."""
#     net_u_first_layer_weights = model.net_u.layers[0].weight.data
#     net_c_first_layer_weights = model.net_c.layers[0].weight.data
    
#     net_u_first_layer_magnitude = torch.norm(net_u_first_layer_weights, dim=0).cpu().numpy()
#     net_c_first_layer_magnitude = torch.norm(net_c_first_layer_weights, dim=0).cpu().numpy()
    
#     known_feature_names = feature_names[:2]  # 'Size' and 'Shape'
#     unknown_feature_names = feature_names[2:]  # 'Unknown Concept 1' and 'Unknown Concept 2'
    
#     x_known = np.arange(len(known_feature_names))
#     x_unknown = np.arange(len(unknown_feature_names))
#     width = 0.35

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#     ax1.bar(x_known - width/2, net_u_first_layer_magnitude[:len(known_feature_names)], width, label='Unknown Concept Network (net_u)')
#     ax1.bar(x_known + width/2, net_c_first_layer_magnitude[:len(known_feature_names)], width, label='Concept Network (net_c)')
#     ax1.set_xlabel('Known Features')
#     ax1.set_ylabel('Magnitude')
#     ax1.set_title(f'Known Feature Magnitudes with EYE Regularization (Alpha={alpha})')
#     ax1.set_xticks(x_known)
#     ax1.set_xticklabels(known_feature_names)
#     ax1.legend()

#     ax2.bar(x_unknown - width/2, net_u_first_layer_magnitude[len(known_feature_names):len(known_feature_names) + len(unknown_feature_names)], width, label='Unknown Concept Network (net_u)')
#     ax2.bar(x_unknown + width/2, net_c_first_layer_magnitude[len(known_feature_names):len(known_feature_names) + len(unknown_feature_names)], width, label='Concept Network (net_c)')
#     ax2.set_xlabel('Unknown Features')
#     ax2.set_ylabel('Magnitude')
#     ax2.set_title(f'Unknown Feature Magnitudes with EYE Regularization (Alpha={alpha})')
#     ax2.set_xticks(x_unknown)
#     ax2.set_xticklabels(unknown_feature_names)
#     ax2.legend()

#     fig.tight_layout()
#     plot_filename = os.path.join(save_dir, f'feature_magnitudes_alpha_{alpha}.png')
#     plt.savefig(plot_filename)
#     plt.close()

    
# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown).to(device)
#     train_losses = []
#     test_losses = []

#     for epoch in range(epochs):
#         model.train()
#         total_train_loss = 0
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 eye_value = eye_regularization(theta_x, theta_c)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)
#         print(f'Epoch {epoch + 1}, Average Train Loss: {average_train_loss}')

#         model.eval()
#         total_test_loss = 0
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)
#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)
#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Average Test Loss: {average_test_loss}')

#     return train_losses, test_losses

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, optimizer = initialize_models(device)
#     batch_size = 32

#     alphas = [(0.001, 0.001), (0.01, 0.01), (0.1, 0.1), (1, 1), (10, 10), (100, 100)]
#     overall_train_losses = []
#     overall_test_losses = []
#     num_shortcuts = 3

#     cases = [
#         {"model": cbm, "include_shortcut": False, "model_name": "CBM", "use_eye": False},
#         {"model": cbm, "include_shortcut": True, "model_name": "CBM", "use_eye": False},
#         {"model": ccm, "include_shortcut": False, "model_name": "CCM", "use_eye": True},
#         {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
#     ]

#     specific_scenario = {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
#     specific_train_losses = []
#     specific_test_losses = []

#     feature_names = ['Size', 'Shape', 'Unknown Concept 1', 'Unknown Concept 2']

#     for alpha_known, alpha_unknown in alphas:
#         results = {}
#         for case in cases:
#             model = case["model"]
#             model_name = case["model_name"]
#             use_eye = case["use_eye"]

#             if use_eye:
#                 model = CCMWithEYE(model.net_c, model.net_u, model.net_y, lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown)
#                 optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters()), lr=0.001)

#             case_label = f"{model_name} {'With' if case['include_shortcut'] else 'Without'} Shortcut, Alpha Known={alpha_known}, Alpha Unknown={alpha_unknown}"
#             print(f"Training case: {case_label}")
#             train_features, train_known_concepts, train_target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=False, num_shortcuts=num_shortcuts)
#             train_dataloader = prepare_dataloaders(train_features, train_known_concepts, train_target, batch_size=batch_size)

#             test_features, test_known_concepts, test_target = generate_dataset(num_samples=200, size_mean=10, include_shortcut=case['include_shortcut'], test_set=True, num_shortcuts=num_shortcuts)
#             test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size=batch_size)

#             plot_save_dir = os.path.join("results", f"alpha_known_{alpha_known}_unknown_{alpha_unknown}", "distributions")
#             compare_distributions(train_features, test_features, alpha_known, plot_save_dir)

#             train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, epochs=10, device=device, EYE_penalty=use_eye)

#             if train_losses and test_losses:
#                 add_results(results, case_label + " - Train", train_losses)
#                 add_results(results, case_label + " - Test", test_losses)

#             if case == specific_scenario:
#                 specific_train_losses.append(np.mean(train_losses))
#                 specific_test_losses.append(np.mean(test_losses))

#         alpha_dir = os.path.join("results", f"alpha_known_{alpha_known}_unknown_{alpha_unknown}")
#         os.makedirs(alpha_dir, exist_ok=True)
#         save_results(results, filename=os.path.join(alpha_dir, "training_results.json"))
#         plot_results(results, save_dir=alpha_dir)

#         if case == specific_scenario:
#             analyze_regularization_effect(model, alpha_unknown, alpha_dir, feature_names)

#     scenario_name = f"{specific_scenario['model_name']}_With_Shortcut"
#     plot_loss_vs_alpha(specific_train_losses, specific_test_losses, [a[0] for a in alphas], save_dir="results", scenario_name=scenario_name)

# if __name__ == "__main__":
#     main()

# # Method 3: Explicitly handle shortcuts.
# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import EYERegularization, cbm_loss
# from eval import add_results, plot_results, save_results, load_results
# from itertools import chain
# import os
# from cycler import cycler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3          # no. of input features (sizes, unknown_concept_1, unknown_concept_2)
#     concept_size = 5        # number of concepts learned by the intermediate layers of the model (dimensionality of the concept space)
#     output_size = 1         # number of output features from the model
#     hidden_dim = 20         # number of units in the hidden layers of the MLP

#     # Separate instances for CBM and CCM
#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

#     # Combine parameters from both models without duplicates
#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s_ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         # preds and targets are tuples
#         concepts_pred, y_pred = preds  # Unpack predictions
#         concepts_label, y_label = targets  # Unpack targets assuming targets come as a tuple already

#         # Ensure y_label is [batch_size, 1] to match y_pred
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         # Calculate loss
#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses, test_losses = [], []

#     for epoch in range(epochs):
#         total_train_loss = 0
#         model.train()
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters()) if hasattr(model, 'net_s') else []
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)
#         print(f'Epoch {epoch + 1}, Training Loss: {average_train_loss}')

#         # Evaluate on test set
#         total_test_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
                
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')

#     return train_losses, test_losses

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32

#     alphas = [0.01]  
#     num_shortcuts = 10

#     cases = [
#         {"model": cbm, "include_shortcut": False, "model_name": "CBM", "use_eye": False},
#         {"model": cbm, "include_shortcut": True, "model_name": "CBM", "use_eye": False},
#         {"model": ccm, "include_shortcut": False, "model_name": "CCM", "use_eye": True},
#         {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
#     ]

#     for alpha in alphas:
#         results = {}
#         for case in cases:
#             model = case["model"]
#             model_name = case["model_name"]
#             use_eye = case["use_eye"]

#             if use_eye:
#                 model = CCMWithEYE(model.net_c, model.net_u, model.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#                 optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#             case_label = f"{model_name} {'With' if case['include_shortcut'] else 'Without'} Shortcut, Alpha={alpha}"
#             print(f"Training case: {case_label}")
#             features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=False, num_shortcuts=num_shortcuts)
#             train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

#             features_test, known_concepts_test, target_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=True, num_shortcuts=num_shortcuts)
#             test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

#             train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=1000, device=device, EYE_penalty=use_eye)

#             if train_losses and test_losses:
#                 add_results(results, case_label, train_losses, test_losses)

#         alpha_dir = os.path.join("results", f"alpha_{alpha}")
#         os.makedirs(alpha_dir, exist_ok=True)
#         save_results(results, filename=os.path.join(alpha_dir, "training_results.json"))
#         plot_results(results, alpha=alpha, save_dir=alpha_dir)

# if __name__ == "__main__":
#     main()


# # # Method 3.2: Explicitly handle shortcuts. Plot Loss vs distrib shift. 
# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import EYERegularization, cbm_loss
# from eval import add_results, plot_results, save_results, plot_loss_vs_shift
# from itertools import chain
# import os
# from cycler import cycler

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3  # input size based on dataset features
#     concept_size = 5
#     output_size = 1
#     hidden_dim = 20

#     # Separate instances for CBM and CCM
#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

#     # Combine parameters from both models without duplicates
#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s_ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         # preds and targets are tuples
#         concepts_pred, y_pred = preds  # Unpack predictions
#         concepts_label, y_label = targets  # Unpack targets assuming targets come as a tuple already

#         # Ensure y_label is [batch_size, 1] to match y_pred
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         # Calculate loss
#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses, test_losses = [], []

#     for epoch in range(epochs):
#         total_train_loss = 0
#         model.train()
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters()) if hasattr(model, 'net_s') else []
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)
#         print(f'Epoch {epoch + 1}, Training Loss: {average_train_loss}')

#         # Evaluate on test set
#         total_test_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
                
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')

#     return train_losses, test_losses

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32

#     alpha = 0.01  
#     num_shortcuts = 10  
#     shift_magnitudes = [0, 1, 2, 3, 4, 5]  

#     results = {}
#     for shift in shift_magnitudes:
#         model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#         optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#         case_label = f"CCM EYE With Shortcut, Alpha={alpha}, Shift={shift}"
#         print(f"Training case: {case_label}")
#         features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=False, num_shortcuts=num_shortcuts, shift_magnitude=0)
#         train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

#         features_test, known_concepts_test, target_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=True, num_shortcuts=num_shortcuts, shift_magnitude=shift)
#         test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

#         train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=50, device=device, EYE_penalty=True)

#         if train_losses and test_losses:
#             add_results(results, case_label, train_losses, test_losses)

#     shift_dir = os.path.join("results", f"shift_experiment_alpha_{alpha}_shortcuts_{num_shortcuts}")
#     os.makedirs(shift_dir, exist_ok=True)
#     save_results(results, filename=os.path.join(shift_dir, "training_results.json"))
#     plot_results(results, alpha=alpha, save_dir=shift_dir)

#     # Plot loss vs shift magnitude
#     plot_loss_vs_shift(results, save_dir=shift_dir)

# if __name__ == "__main__":
#     main()

# # # # Method 3.3: Explicitly handle shortcuts. Track weights.
# import torch
# import numpy as np
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import EYERegularization, cbm_loss
# from eval import add_results, plot_results, save_results, plot_loss_vs_shift, plot_feature_correlation
# from itertools import chain
# import os
# from cycler import cycler
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3  # input size based on dataset features
#     concept_size = 5
#     output_size = 1
#     hidden_dim = 20

#     # Separate instances for CBM and CCM
#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

#     # Combine parameters from both models without duplicates
#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s_ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         # preds and targets are tuples
#         concepts_pred, y_pred = preds  # Unpack predictions
#         concepts_label, y_label = targets  # Unpack targets assuming targets come as a tuple already

#         # Ensure y_label is [batch_size, 1] to match y_pred
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         # Calculate loss
#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss


# def plot_weight_magnitudes(weight_magnitudes, save_dir):
#     """Plot weight magnitudes for each concept over epochs."""
#     plt.figure(figsize=(12, 8))
#     for concept, magnitudes in weight_magnitudes.items():
#         plt.plot(magnitudes, label=concept)
#     plt.title('Weight Magnitudes Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Weight Magnitude')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'weight_magnitudes.png'))
#     else:
#         plt.show()

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses, test_losses = [], []

#     weight_magnitudes = {'size': [], 'unknown_concept_1': [], 'unknown_concept_2': []}

#     for epoch in range(epochs):
#         total_train_loss = 0
#         model.train()
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters()) if hasattr(model, 'net_s') else []
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)
#         print(f'Epoch {epoch + 1}, Training Loss: {average_train_loss}')

#         # Evaluate on test set
#         total_test_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
                
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')

#         # Track weight magnitudes for each concept
#         net_c_weights = model.net_c.layers[0].weight.detach().cpu().numpy()
#         net_u_weights = model.net_u.layers[0].weight.detach().cpu().numpy()
#         net_s_weights = model.net_s.layers[0].weight.detach().cpu().numpy() if hasattr(model, 'net_s') else []

#         weight_magnitudes['size'].append(np.mean(np.abs(net_c_weights[:, 0])))
#         weight_magnitudes['unknown_concept_1'].append(np.mean(np.abs(net_u_weights[:, 1])))
#         weight_magnitudes['unknown_concept_2'].append(np.mean(np.abs(net_u_weights[:, 2])))

#     return train_losses, test_losses, weight_magnitudes

 
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32

#     alpha = 0.01  
#     num_shortcuts = 2
#     shift_magnitudes = [0, 1, 2, 3, 4, 5]  

#     results = {}
#     weight_magnitudes_all_shifts = []
#     for shift in shift_magnitudes:
#         model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#         optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#         case_label = f"CCM EYE With Shortcut, Alpha={alpha}, Shift={shift}"
#         print(f"Training case: {case_label}")
#         features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=False, num_shortcuts=num_shortcuts, shift_magnitude=0)
#         train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

#         features_test, known_concepts_test, target_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=True, num_shortcuts=num_shortcuts, shift_magnitude=shift)
#         test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

#         train_losses, test_losses, weight_magnitudes = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=1000, device=device, EYE_penalty=True)
#         weight_magnitudes_all_shifts.append((shift, weight_magnitudes))

#         if train_losses and test_losses:
#             add_results(results, case_label, train_losses, test_losses)

#         shift_specific_dir = os.path.join("results", f"shift_{shift}")
#         os.makedirs(shift_specific_dir, exist_ok=True)
#         plot_feature_correlation(features, target, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Train Shift={shift}")
#         plot_feature_correlation(features_test, target_test, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Test Shift={shift}")

#     shift_dir = os.path.join("results", f"shift_experiment_alpha_{alpha}_shortcuts_{num_shortcuts}")
#     os.makedirs(shift_dir, exist_ok=True)
#     save_results(results, filename=os.path.join(shift_dir, "training_results.json"))
#     plot_results(results, alpha=alpha, save_dir=shift_dir)
    
#     # Plot weight magnitudes
#     for shift, weight_magnitudes in weight_magnitudes_all_shifts:
#         shift_specific_dir = os.path.join(shift_dir, f"shift_{shift}")
#         plot_weight_magnitudes(weight_magnitudes, save_dir=shift_specific_dir)

#     # Plot loss vs shift magnitude
#     plot_loss_vs_shift(results, save_dir=shift_dir)

# if __name__ == "__main__":
#     main()

# # # # Method 3.4: Explicitly handle shortcuts. Contour.
# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import EYERegularization, cbm_loss
# from eval import add_results, plot_results, save_results, plot_loss_vs_shift, plot_feature_correlation
# from contour_plots import generate_contour_plot
# from itertools import chain
# import os
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3  # input size based on dataset features
#     concept_size = 5
#     output_size = 1
#     hidden_dim = 20

#     # Separate instances for CBM and CCM
#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

#     # Combine parameters from both models without duplicates
#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s_ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         # preds and targets are tuples
#         concepts_pred, y_pred = preds  # Unpack predictions
#         concepts_label, y_label = targets  # Unpack targets assuming targets come as a tuple already

#         # Ensure y_label is [batch_size, 1] to match y_pred
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         # Calculate loss
#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss

# def plot_weight_magnitudes(weight_magnitudes, save_dir):
#     """Plot weight magnitudes for each concept over epochs."""
#     plt.figure(figsize=(12, 8))
#     for concept, magnitudes in weight_magnitudes.items():
#         plt.plot(magnitudes, label=concept)
#     plt.title('Weight Magnitudes Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Weight Magnitude')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'weight_magnitudes.png'))
#     else:
#         plt.show()

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses = []
#     test_losses = []
#     weight_magnitudes = {'Size': [], 'Unknown Concept 1': [], 'Unknown Concept 2': []}

#     for epoch in range(epochs):
#         total_loss = 0
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters())
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         average_loss = total_loss / len(train_dataloader)
#         train_losses.append(average_loss)
#         print(f'Epoch {epoch + 1}, Train Loss: {average_loss}')

#         # Evaluate on test set
#         model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
#                 test_loss += loss.item()

#         average_test_loss = test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')
#         model.train()

#         # Track weight magnitudes
#         weight_magnitudes['Size'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 0])).item())
#         weight_magnitudes['Unknown Concept 1'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 1])).item())
#         weight_magnitudes['Unknown Concept 2'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 2])).item())

#     return train_losses, test_losses, weight_magnitudes

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32

#     alpha = 0.01  # Set the specific alpha value
#     num_shortcuts = 3  # Set the number of shortcuts
#     shift_magnitudes = [0]  # Define shift magnitudes

#     results = {}
#     weight_magnitudes_all_shifts = []
#     for shift in shift_magnitudes:
#         model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#         optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#         case_label = f"CCM EYE With Shortcut, Alpha={alpha}, Shift={shift}"
#         print(f"Training case: {case_label}")
#         features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=False, num_shortcuts=num_shortcuts, shift_magnitude=0)
#         train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

#         features_test, known_concepts_test, target_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=True, num_shortcuts=num_shortcuts, shift_magnitude=shift)
#         test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

#         train_losses, test_losses, weight_magnitudes = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=1000, device=device, EYE_penalty=True)
#         weight_magnitudes_all_shifts.append((shift, weight_magnitudes))

#         if train_losses and test_losses:
#             add_results(results, case_label, train_losses, test_losses)

#         shift_specific_dir = os.path.join("results", f"shift_{shift}")
#         os.makedirs(shift_specific_dir, exist_ok=True)
#         plot_feature_correlation(features, target, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Train Shift={shift}")
#         plot_feature_correlation(features_test, target_test, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Test Shift={shift}")

#     shift_dir = os.path.join("results", f"shift_experiment_alpha_{alpha}_shortcuts_{num_shortcuts}")
#     os.makedirs(shift_dir, exist_ok=True)
#     save_results(results, filename=os.path.join(shift_dir, "training_results.json"))
#     plot_results(results, alpha=alpha, save_dir=shift_dir)
    
#     # Plot weight magnitudes
#     for shift, weight_magnitudes in weight_magnitudes_all_shifts:
#         shift_specific_dir = os.path.join(shift_dir, f"shift_{shift}")
#         plot_weight_magnitudes(weight_magnitudes, save_dir=shift_specific_dir)

#     # Plot loss vs shift magnitude
#     plot_loss_vs_shift(results, save_dir=shift_dir)

#     # Generate contour plots for the EYE regularization term
#     # generate_contour_plot(lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha, save_dir=shift_dir)
#     generate_contour_plot(0.01, 0.01, 0.01, 'results')
# if __name__ == "__main__":
#     main()

# # # # Method 3.5: Explicitly handle shortcuts. Completeness score.
# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import cbm_loss, EYERegularization
# from eval import add_results, plot_results, save_results, plot_feature_correlation, plot_loss_vs_shift, plot_weight_magnitudes
# from contour_plots import generate_contour_plot
# from itertools import chain
# import os
# import numpy as np
# import json

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3  # input size based on dataset features
#     concept_size = 5
#     output_size = 1
#     hidden_dim = 20

#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u, net_y=net_y)

#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         concepts_pred, y_pred = preds
#         concepts_label, y_label = targets

#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses = []
#     test_losses = []
#     weight_magnitudes = []

#     for epoch in range(epochs):
#         total_train_loss = 0
#         model.train()
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters())
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         average_train_loss = total_train_loss / len(train_dataloader)
#         train_losses.append(average_train_loss)

#         total_test_loss = 0
#         model.eval()
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
#                 total_test_loss += loss.item()

#         average_test_loss = total_test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)

#         # Track weight magnitudes
#         with torch.no_grad():
#             known_weight_magnitude = torch.norm(torch.cat([param.view(-1) for param in model.net_c.parameters()]), p=1).item()
#             unknown_weight_magnitude = torch.norm(torch.cat([param.view(-1) for param in model.net_u.parameters()]), p=1).item()
#             shortcut_weight_magnitude = torch.norm(torch.cat([param.view(-1) for param in model.net_s.parameters()]), p=1).item()
#             weight_magnitudes.append((known_weight_magnitude, unknown_weight_magnitude, shortcut_weight_magnitude))

#         print(f'Epoch {epoch + 1}, Train Loss: {average_train_loss}, Test Loss: {average_test_loss}')

#     return train_losses, test_losses, weight_magnitudes

# def completeness_score(before_removal, after_removal):
#     return np.mean(before_removal) - np.mean(after_removal)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32

#     alphas = [0.001, 0.01]
#     shift_strengths = [0]
#     num_shortcuts = 10  # Number of shortcuts to introduce

#     completeness_scores_known = []
#     completeness_scores_unknown = []

#     for alpha in alphas:
#         results_correlate_known = {}
#         results_correlate_unknown = {}
#         weight_magnitudes_all_shifts_known = {shift: [] for shift in shift_strengths}
#         weight_magnitudes_all_shifts_unknown = {shift: [] for shift in shift_strengths}

#         for shift in shift_strengths:
#             for correlate_with_known in [True, False]:
#                 train_features, train_known_concepts, train_target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=False, test_set=False, num_shortcuts=num_shortcuts, correlate_with_known=correlate_with_known)
#                 test_features, test_known_concepts, test_target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, size_shift=True, test_set=True, num_shortcuts=num_shortcuts, correlate_with_known=correlate_with_known)

#                 train_dataloader = prepare_dataloaders(train_features, train_known_concepts, train_target, batch_size=batch_size)
#                 test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size=batch_size)

#                 model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#                 optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#                 scenario_label = "Correlate_with_Known" if correlate_with_known else "Correlate_with_Unknown"
#                 case_label = f"{scenario_label}, Shift={shift}, Alpha={alpha}"
#                 print(f"Training case: {case_label}")
                
#                 train_losses, test_losses, weight_magnitudes = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=30, device=device, EYE_penalty=True)

#                 if correlate_with_known:
#                     add_results(results_correlate_known, case_label, train_losses, test_losses)
#                     weight_magnitudes_all_shifts_known[shift].append((alpha, weight_magnitudes))
#                 else:
#                     add_results(results_correlate_unknown, case_label, train_losses, test_losses)
#                     weight_magnitudes_all_shifts_unknown[shift].append((alpha, weight_magnitudes))

#         # Calculate completeness score
#         try:
#             completeness_score_known = completeness_score(results_correlate_known[f"Correlate_with_Known, Shift={shift_strengths[0]}, Alpha={alpha}"]["train_losses"], results_correlate_known[f"Correlate_with_Known, Shift={max(shift_strengths)}, Alpha={alpha}"]["train_losses"])
#             completeness_scores_known.append(completeness_score_known)

#             completeness_score_unknown = completeness_score(results_correlate_unknown[f"Correlate_with_Unknown, Shift={shift_strengths[0]}, Alpha={alpha}"]["train_losses"], results_correlate_unknown[f"Correlate_with_Unknown, Shift={max(shift_strengths)}, Alpha={alpha}"]["train_losses"])
#             completeness_scores_unknown.append(completeness_score_unknown)
#         except KeyError as e:
#             print(f"KeyError: {e}")
#             print("Skipping this alpha value due to missing data.")

#         # Save results and plots
#         alpha_dir = os.path.join("results", f"alpha_{alpha}")
#         os.makedirs(alpha_dir, exist_ok=True)
#         save_results(results_correlate_known, filename=os.path.join(alpha_dir, "training_results_correlate_known.json"))
#         save_results(results_correlate_unknown, filename=os.path.join(alpha_dir, "training_results_correlate_unknown.json"))

#         plot_loss_vs_shift(results_correlate_known, shift_strengths, scenario="Correlate_with_Known", save_dir=alpha_dir)
#         plot_loss_vs_shift(results_correlate_unknown, shift_strengths, scenario="Correlate_with_Unknown", save_dir=alpha_dir)

#         plot_weight_magnitudes(weight_magnitudes_all_shifts_known, alphas, scenario="Correlate_with_Known", save_dir=alpha_dir)
#         plot_weight_magnitudes(weight_magnitudes_all_shifts_unknown, alphas, scenario="Correlate_with_Unknown", save_dir=alpha_dir)

#     # Save completeness scores
#     completeness_scores = {
#         "Correlate_with_Known": completeness_scores_known,
#         "Correlate_with_Unknown": completeness_scores_unknown
#     }
#     with open(os.path.join("results", "completeness_scores.json"), 'w') as f:
#         json.dump(completeness_scores, f)

# if __name__ == "__main__":
#     main()
# import torch
# from torch import optim, nn
# from models import CCM, CBM, MLP, CCMWithEYE
# from utils import prepare_dataloaders, generate_dataset
# from regularisation import EYERegularization, cbm_loss
# from eval import add_results, plot_results, save_results, plot_loss_vs_shift, plot_feature_correlation
# from contour_plots import generate_contour_plot
# from itertools import chain
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def initialize_models(device):
#     input_size = 3  # input size based on dataset features
#     concept_size = 5
#     output_size = 1
#     hidden_dim = 20

#     net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
#     net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
#     combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

#     cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
#     ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

#     optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
#     return cbm, ccm, net_s_ccm, optimizer

# class CBMLoss(nn.Module):
#     def __init__(self, lambda_concept=1):
#         super(CBMLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.lambda_concept = lambda_concept

#     def forward(self, preds, targets):
#         concepts_pred, y_pred = preds
#         concepts_label, y_label = targets
#         y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

#         primary_loss = self.mse_loss(y_pred, y_label)
#         concept_loss = self.mse_loss(concepts_pred, concepts_label)
#         return primary_loss + self.lambda_concept * concept_loss

# def plot_weight_magnitudes(weight_magnitudes, save_dir):
#     plt.figure(figsize=(12, 8))
#     for concept, magnitudes in weight_magnitudes.items():
#         plt.plot(magnitudes, label=concept)
#     plt.title('Weight Magnitudes Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Weight Magnitude')
#     plt.legend()
#     plt.grid(True)
    
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         plt.savefig(os.path.join(save_dir, 'weight_magnitudes.png'))
#     else:
#         plt.show()

# def completeness_score(model, dataloader, device):
#     model.eval()
#     total_loss = 0
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             concepts_pred, y_pred = model(data)
#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
#             loss = criterion(y_pred, y_label)
#             total_loss += loss.item()
#     return total_loss / len(dataloader)

# def analyze_weights(model, known_concepts_size, unknown_concepts_size):
#     known_weights = []
#     unknown_weights = []

#     for name, param in model.named_parameters():
#         if "net_c" in name:
#             known_weights.append(param.detach().cpu().numpy().flatten())
#         elif "net_u" in name:
#             unknown_weights.append(param.detach().cpu().numpy().flatten())

#     known_weights = np.concatenate(known_weights)
#     unknown_weights = np.concatenate(unknown_weights)

#     known_weight_magnitude = np.mean(np.abs(known_weights))
#     unknown_weight_magnitude = np.mean(np.abs(unknown_weights))

#     return known_weight_magnitude, unknown_weight_magnitude

# def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model.to(device)
#     model.train()
#     loss_func = CBMLoss(lambda_concept=1).to(device)
#     eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
#     train_losses = []
#     test_losses = []
#     weight_magnitudes = {'Size': [], 'Unknown Concept 1': [], 'Unknown Concept 2': []}
#     weight_analysis = {'Known Weights': [], 'Unknown Weights': []}

#     for epoch in range(epochs):
#         total_loss = 0
#         for data, target in train_dataloader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
            
#             outputs = model(data)
#             concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#             concepts_label = target[:, :-1]
#             y_label = target[:, -1].unsqueeze(1)
            
#             loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
#             if EYE_penalty:
#                 theta_x = list(model.net_u.parameters())
#                 theta_c = list(model.net_c.parameters())
#                 theta_s = list(model.net_s.parameters())
#                 eye_value = eye_regularization(theta_x, theta_c, theta_s)
#                 loss += eye_value

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         average_loss = total_loss / len(train_dataloader)
#         train_losses.append(average_loss)
#         print(f'Epoch {epoch + 1}, Train Loss: {average_loss}')

#         model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for data, target in test_dataloader:
#                 data, target = data.to(device), target.to(device)
#                 outputs = model(data)
#                 concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

#                 concepts_label = target[:, :-1]
#                 y_label = target[:, -1].unsqueeze(1)

#                 loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
#                 test_loss += loss.item()

#         average_test_loss = test_loss / len(test_dataloader)
#         test_losses.append(average_test_loss)
#         print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')
#         model.train()

#         known_weight_magnitude, unknown_weight_magnitude = analyze_weights(model, known_concepts_size=2, unknown_concepts_size=2)
#         weight_analysis['Known Weights'].append(known_weight_magnitude)
#         weight_analysis['Unknown Weights'].append(unknown_weight_magnitude)

#         weight_magnitudes['Size'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 0])).item())
#         weight_magnitudes['Unknown Concept 1'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 1])).item())
#         weight_magnitudes['Unknown Concept 2'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 2])).item())

#     return train_losses, test_losses, weight_magnitudes, weight_analysis

# def save_model(model, path):
#     torch.save(model.state_dict(), path)

# def load_model(model, path, device):
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     return model

# def train_and_evaluate():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cbm, ccm, net_s, optimizer = initialize_models(device)
#     batch_size = 32
#     epochs = 1000
#     alpha = 0.01
#     num_shortcuts = 5
#     results = {}
#     completeness_scores = {'C': [], 'U': []}

#     for shortcut_type in ['C', 'U']:
#         for shift in [0, 3]:
#             model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
#             optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

#             case_label = f"CCM EYE With Shortcut {shortcut_type}, Alpha={alpha}, Shift={shift}"
#             print(f"Training case: {case_label}")

#             features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=False, num_shortcuts=num_shortcuts, shift_magnitude=0, shortcut_type=shortcut_type)
#             train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

#             features_test, known_concepts_test, target_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=True, num_shortcuts=num_shortcuts, shift_magnitude=shift, shortcut_type=shortcut_type)
#             test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

#             # Evaluate completeness score before training
#             initial_completeness_score = completeness_score(model, test_dataloader, device)

#             train_losses, test_losses, weight_magnitudes, weight_analysis = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=epochs, device=device, EYE_penalty=True)

#             # Evaluate completeness score after training
#             final_completeness_score = completeness_score(model, test_dataloader, device)

#             completeness_scores[shortcut_type].append({
#                 'initial': initial_completeness_score,
#                 'final': final_completeness_score
#             })

#             if train_losses and test_losses:
#                 add_results(results, case_label, train_losses, test_losses)

#             shift_specific_dir = os.path.join("results", f"shift_{shift}")
#             os.makedirs(shift_specific_dir, exist_ok=True)
#             plot_feature_correlation(features, target, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Train Shift={shift}")
#             plot_feature_correlation(features_test, target_test, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Test Shift={shift}")

#     shift_dir = os.path.join("results", f"shift_experiment_alpha_{alpha}_shortcuts_{num_shortcuts}")
#     os.makedirs(shift_dir, exist_ok=True)
#     save_results(results, filename=os.path.join(shift_dir, "training_results.json"))
#     plot_results(results, alpha=alpha, save_dir=shift_dir)

#     for shortcut_type in ['C', 'U']:
#         initial_scores = [x['initial'] for x in completeness_scores[shortcut_type]]
#         final_scores = [x['final'] for x in completeness_scores[shortcut_type]]
#         plt.figure(figsize=(12, 8))
#         plt.plot(initial_scores, label=f'Initial {shortcut_type}')
#         plt.plot(final_scores, label=f'Final {shortcut_type}')
#         plt.title(f'Completeness Score for Shortcut Type {shortcut_type}')
#         plt.xlabel('Shift Magnitude')
#         plt.ylabel('Completeness Score')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(shift_dir, f'completeness_score_{shortcut_type}.png'))
#         plt.show()

#     plot_loss_vs_shift(results, save_dir=shift_dir)
#     generate_contour_plot(0.01, 0.01, 0.01, 'results')

# if __name__ == "__main__":
#     train_and_evaluate()

import torch
from torch import optim, nn
from models import CCM, CBM, MLP, CCMWithEYE
from utils import prepare_dataloaders, generate_dataset
from regularisation import EYERegularization, cbm_loss
from eval import add_results, plot_results, save_results, plot_loss_vs_shift, plot_feature_correlation
from contour_plots import generate_contour_plot
from itertools import chain
import os
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def completeness_score(model, dataloader, device):
    model.eval()
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            concepts_pred, y_pred = model(data)
            total_score += torch.mean((concepts_pred - target[:, :-1]).abs()).item()
            count += 1
    return total_score / count

def plot_shortcut_correlation(shortcuts, model, dataloader, save_dir, set_name):
    model.eval()
    all_preds = []
    all_shortcuts = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            _, y_pred = model(data)
            all_preds.append(y_pred.cpu().numpy())
            all_shortcuts.append(shortcuts[:len(y_pred)])  # Adjusting the shortcuts length to match predictions

    all_preds = np.concatenate(all_preds).squeeze()
    all_shortcuts = np.concatenate(all_shortcuts).squeeze()
    
    correlation = np.corrcoef(all_preds, all_shortcuts)[0, 1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(all_shortcuts, all_preds, alpha=0.5)
    plt.xlabel('Shortcut Presence')
    plt.ylabel('Model Prediction')
    plt.title(f'Shortcut Correlation with Model Prediction ({set_name})')
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'shortcut_correlation_{set_name}.png'))
    
    return correlation

def plot_shortcut_correlation_all(shortcut_correlations_all, alpha_values, save_dir, shortcut_type):
    plt.figure(figsize=(10, 6))
    for shortcut_correlations, alpha in zip(shortcut_correlations_all, alpha_values):
        shifts = [sc['shift'] for sc in shortcut_correlations]
        initial_corrs = [sc['initial'] for sc in shortcut_correlations]
        final_corrs = [sc['final'] for sc in shortcut_correlations]

        plt.plot(shifts, initial_corrs, label=f'Initial Alpha={alpha}')
        plt.plot(shifts, final_corrs, label=f'Final Alpha={alpha}')

    plt.xlabel('Shift Magnitude')
    plt.ylabel('Shortcut Correlation')
    plt.title(f'Shortcut Correlation for Shortcut Type {shortcut_type}')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'shortcut_correlation_{shortcut_type}.png'))


def plot_completeness_vs_shift(completeness_scores, save_dir, shortcut_type):
    shifts = [cs['shift'] for cs in completeness_scores]
    initial_scores = [cs['initial'] for cs in completeness_scores]
    final_scores = [cs['final'] for cs in completeness_scores]
    
    plt.figure(figsize=(10, 6))
    plt.plot(shifts, initial_scores, label=f'Initial {shortcut_type}')
    plt.plot(shifts, final_scores, label=f'Final {shortcut_type}')
    plt.xlabel('Shift Magnitude')
    plt.ylabel('Completeness Score')
    plt.title(f'Completeness Score for Shortcut Type {shortcut_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir)

def plot_completeness_vs_shift_all(completeness_scores_all, alpha_values, save_dir, shortcut_type):
    plt.figure(figsize=(10, 6))
    for completeness_scores, alpha in zip(completeness_scores_all, alpha_values):
        shifts = [cs['shift'] for cs in completeness_scores]
        initial_scores = [cs['initial'] for cs in completeness_scores]
        final_scores = [cs['final'] for cs in completeness_scores]

        plt.plot(shifts, initial_scores, label=f'Initial Alpha={alpha}')
        plt.plot(shifts, final_scores, label=f'Final Alpha={alpha}')

    plt.xlabel('Shift Magnitude')
    plt.ylabel('Completeness Score')
    plt.title(f'Completeness Score for Shortcut Type {shortcut_type}')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'completeness_score_{shortcut_type}.png'))


def initialize_models(device):
    input_size = 3
    concept_size = 5
    output_size = 1
    hidden_dim = 20

    net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_y_cbm = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    
    net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_u_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_s_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    combined_net_y = MLP(input_dim=3 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

    cbm = CBM(net_c=net_c_cbm, net_y=net_y_cbm)
    ccm = CCM(net_c=net_c_ccm, net_u=net_u_ccm, net_y=combined_net_y)

    optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters(), net_s_ccm.parameters()), lr=0.001)
    
    return cbm, ccm, net_s_ccm, optimizer

class CBMLoss(nn.Module):
    def __init__(self, lambda_concept=1):
        super(CBMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_concept = lambda_concept

    def forward(self, preds, targets):
        concepts_pred, y_pred = preds
        concepts_label, y_label = targets

        y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

        primary_loss = self.mse_loss(y_pred, y_label)
        concept_loss = self.mse_loss(concepts_pred, concepts_label)
        return primary_loss + self.lambda_concept * concept_loss

def plot_weight_magnitudes(weight_magnitudes, save_dir):
    plt.figure(figsize=(12, 8))
    for concept, magnitudes in weight_magnitudes.items():
        plt.plot(magnitudes, label=concept)
    plt.title('Weight Magnitudes Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Weight Magnitude')
    plt.legend()
    plt.grid(True)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'weight_magnitudes.png'))
    else:
        plt.show()

def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, alpha_shortcut, epochs=10, device=None, EYE_penalty=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    loss_func = CBMLoss(lambda_concept=1).to(device)
    eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown, lambda_eye_shortcut=alpha_shortcut).to(device)
    train_losses = []
    test_losses = []
    weight_magnitudes = {'Size': [], 'Unknown Concept 1': [], 'Unknown Concept 2': []}

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

            concepts_label = target[:, :-1]
            y_label = target[:, -1].unsqueeze(1)
            
            loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            
            if EYE_penalty:
                theta_x = list(model.net_u.parameters())
                theta_c = list(model.net_c.parameters())
                theta_s = list(model.net_s.parameters())
                eye_value = eye_regularization(theta_x, theta_c, theta_s)
                loss += eye_value

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {average_loss}')

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)

                concepts_label = target[:, :-1]
                y_label = target[:, -1].unsqueeze(1)

                loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
                test_loss += loss.item()

        average_test_loss = test_loss / len(test_dataloader)
        test_losses.append(average_test_loss)
        print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss}')
        model.train()

        weight_magnitudes['Size'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 0])).item())
        weight_magnitudes['Unknown Concept 1'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 1])).item())
        weight_magnitudes['Unknown Concept 2'].append(torch.mean(torch.abs(model.net_u.layers[0].weight.data[:, 2])).item())

    return train_losses, test_losses, weight_magnitudes

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm, ccm, net_s, optimizer = initialize_models(device)
    batch_size = 32
    epochs = 1000
    alpha_values = [0.001, 0.01, 0.1, 1.0]
    num_shortcuts = 10
    results = {}

    for shortcut_type in ['C', 'U']:
        completeness_scores_all = []
        shortcut_correlations_all = []

        for alpha in alpha_values:
            completeness_scores = []
            shortcut_correlations = []

            for shift in [0, 1, 2, 3, 5]:
                model = CCMWithEYE(ccm.net_c, ccm.net_u, ccm.net_y, net_s, lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)
                optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters(), model.net_s.parameters()), lr=0.001)

                case_label = f"CCM EYE With Shortcut {shortcut_type}, Alpha={alpha}, Shift={shift}"
                print(f"Training case: {case_label}")

                features, known_concepts, target, shortcuts = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=False, num_shortcuts=num_shortcuts, shift_magnitude=shift, shortcut_type=shortcut_type)
                train_dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)

                features_test, known_concepts_test, target_test, shortcuts_test = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=True, test_set=True, num_shortcuts=num_shortcuts, shift_magnitude=shift, shortcut_type=shortcut_type)
                test_dataloader = prepare_dataloaders(features_test, known_concepts_test, target_test, batch_size=batch_size)

                initial_completeness_score = completeness_score(model, test_dataloader, device)
                initial_shortcut_corr = plot_shortcut_correlation(shortcuts_test, model, test_dataloader, save_dir=f'results/initial_{shortcut_type}_alpha_{alpha}', set_name='Initial')

                train_losses, test_losses, weight_magnitudes = train(model, train_dataloader, test_dataloader, optimizer, alpha_known=alpha, alpha_unknown=alpha, alpha_shortcut=alpha, epochs=epochs, device=device, EYE_penalty=True)

                final_completeness_score = completeness_score(model, test_dataloader, device)
                final_shortcut_corr = plot_shortcut_correlation(shortcuts_test, model, test_dataloader, save_dir=f'results/final_{shortcut_type}_alpha_{alpha}', set_name='Final')

                completeness_scores.append({
                    'shift': shift,
                    'initial': initial_completeness_score,
                    'final': final_completeness_score
                })

                shortcut_correlations.append({
                    'shift': shift,
                    'initial': initial_shortcut_corr,
                    'final': final_shortcut_corr
                })

                add_results(results, case_label, train_losses, test_losses)

                shift_specific_dir = os.path.join("results", f"shift_{shift}_alpha_{alpha}")
                os.makedirs(shift_specific_dir, exist_ok=True)
                plot_feature_correlation(features, target, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Train Shift={shift}")
                plot_feature_correlation(features_test, target_test, ['Size', 'Unknown Concept 1', 'Unknown Concept 2'], save_dir=shift_specific_dir, set_name=f"Test Shift={shift}")

            completeness_scores_all.append(completeness_scores)
            shortcut_correlations_all.append(shortcut_correlations)

        plot_completeness_vs_shift_all(completeness_scores_all, alpha_values, save_dir=f'results/completeness_score_{shortcut_type}', shortcut_type=shortcut_type)
        plot_shortcut_correlation_all(shortcut_correlations_all, alpha_values, save_dir=f'results/shortcut_correlation_{shortcut_type}', shortcut_type=shortcut_type)

    overall_results_dir = os.path.join("results", "overall")
    os.makedirs(overall_results_dir, exist_ok=True)
    save_results(results, filename=os.path.join(overall_results_dir, "training_results.json"))
    plot_results(results, save_dir=overall_results_dir)

if __name__ == "__main__":
    train_and_evaluate()
