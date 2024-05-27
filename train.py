import torch
from torch import optim, nn
from models import CCM, CBM, MLP, EYERegularization, CCMWithEYE
from utils import prepare_dataloaders, generate_dataset, compare_distributions
from regularisation import EYE, cbm_loss
from eval import add_results, plot_results, save_results, plot_loss_vs_alpha
from itertools import chain
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models(device):
    input_size = 3
    concept_size = 5
    output_size = 1
    hidden_dim = 20

    net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_u = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_y = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    combined_net_y = MLP(input_dim=2 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

    cbm = CBM(net_c=net_c_cbm, net_y=net_y)
    ccm = CCM(net_c=net_c_ccm, net_u=net_u, net_y=combined_net_y)

    optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters()), lr=0.001)
    
    return cbm, ccm, optimizer

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


def evaluate(model, dataloader, loss_func, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)
            concepts_label = target[:, :-1]
            y_label = target[:, -1].unsqueeze(1)
            loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def analyze_regularization_effect(model, alpha, save_dir, feature_names):
    """Analyze the impact of EYE regularization on the model's parameters."""
    net_u_first_layer_weights = model.net_u.layers[0].weight.data
    net_c_first_layer_weights = model.net_c.layers[0].weight.data
    
    net_u_first_layer_magnitude = torch.norm(net_u_first_layer_weights, dim=0).cpu().numpy()
    net_c_first_layer_magnitude = torch.norm(net_c_first_layer_weights, dim=0).cpu().numpy()
    
    known_feature_names = feature_names[:2]  # 'Size' and 'Shape'
    unknown_feature_names = feature_names[2:]  # 'Unknown Concept 1' and 'Unknown Concept 2'
    
    x_known = np.arange(len(known_feature_names))
    x_unknown = np.arange(len(unknown_feature_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(x_known - width/2, net_u_first_layer_magnitude[:len(known_feature_names)], width, label='Unknown Concept Network (net_u)')
    ax1.bar(x_known + width/2, net_c_first_layer_magnitude[:len(known_feature_names)], width, label='Concept Network (net_c)')
    ax1.set_xlabel('Known Features')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'Known Feature Magnitudes with EYE Regularization (Alpha={alpha})')
    ax1.set_xticks(x_known)
    ax1.set_xticklabels(known_feature_names)
    ax1.legend()

    ax2.bar(x_unknown - width/2, net_u_first_layer_magnitude[len(known_feature_names):len(known_feature_names) + len(unknown_feature_names)], width, label='Unknown Concept Network (net_u)')
    ax2.bar(x_unknown + width/2, net_c_first_layer_magnitude[len(known_feature_names):len(known_feature_names) + len(unknown_feature_names)], width, label='Concept Network (net_c)')
    ax2.set_xlabel('Unknown Features')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'Unknown Feature Magnitudes with EYE Regularization (Alpha={alpha})')
    ax2.set_xticks(x_unknown)
    ax2.set_xticklabels(unknown_feature_names)
    ax2.legend()

    fig.tight_layout()
    plot_filename = os.path.join(save_dir, f'feature_magnitudes_alpha_{alpha}.png')
    plt.savefig(plot_filename)
    plt.close()

    
def train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, epochs=10, device=None, EYE_penalty=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    loss_func = CBMLoss(lambda_concept=1).to(device)
    eye_regularization = EYERegularization(lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown).to(device)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
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
                eye_value = eye_regularization(theta_x, theta_c)
                loss += eye_value

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)
        print(f'Epoch {epoch + 1}, Average Train Loss: {average_train_loss}')

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                concepts_pred, y_pred = outputs if len(outputs) == 2 else (outputs, outputs)
                concepts_label = target[:, :-1]
                y_label = target[:, -1].unsqueeze(1)
                loss = loss_func((concepts_pred, y_pred), (concepts_label, y_label))
                total_test_loss += loss.item()

        average_test_loss = total_test_loss / len(test_dataloader)
        test_losses.append(average_test_loss)
        print(f'Epoch {epoch + 1}, Average Test Loss: {average_test_loss}')

    return train_losses, test_losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm, ccm, optimizer = initialize_models(device)
    batch_size = 32

    alphas = [(0.001, 0.001), (0.01, 0.01), (0.1, 0.1), (1, 1), (10, 10), (100, 100)]
    overall_train_losses = []
    overall_test_losses = []
    num_shortcuts = 3

    cases = [
        {"model": cbm, "include_shortcut": False, "model_name": "CBM", "use_eye": False},
        {"model": cbm, "include_shortcut": True, "model_name": "CBM", "use_eye": False},
        {"model": ccm, "include_shortcut": False, "model_name": "CCM", "use_eye": True},
        {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
    ]

    specific_scenario = {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
    specific_train_losses = []
    specific_test_losses = []

    feature_names = ['Size', 'Shape', 'Unknown Concept 1', 'Unknown Concept 2']

    for alpha_known, alpha_unknown in alphas:
        results = {}
        for case in cases:
            model = case["model"]
            model_name = case["model_name"]
            use_eye = case["use_eye"]

            if use_eye:
                model = CCMWithEYE(model.net_c, model.net_u, model.net_y, lambda_eye_known=alpha_known, lambda_eye_unknown=alpha_unknown)
                optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters()), lr=0.001)

            case_label = f"{model_name} {'With' if case['include_shortcut'] else 'Without'} Shortcut, Alpha Known={alpha_known}, Alpha Unknown={alpha_unknown}"
            print(f"Training case: {case_label}")
            train_features, train_known_concepts, train_target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=False, num_shortcuts=num_shortcuts)
            train_dataloader = prepare_dataloaders(train_features, train_known_concepts, train_target, batch_size=batch_size)

            test_features, test_known_concepts, test_target = generate_dataset(num_samples=200, size_mean=10, include_shortcut=case['include_shortcut'], test_set=True, num_shortcuts=num_shortcuts)
            test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size=batch_size)

            plot_save_dir = os.path.join("results", f"alpha_known_{alpha_known}_unknown_{alpha_unknown}", "distributions")
            compare_distributions(train_features, test_features, alpha_known, plot_save_dir)

            train_losses, test_losses = train(model, train_dataloader, test_dataloader, optimizer, alpha_known, alpha_unknown, epochs=10, device=device, EYE_penalty=use_eye)

            if train_losses and test_losses:
                add_results(results, case_label + " - Train", train_losses)
                add_results(results, case_label + " - Test", test_losses)

            if case == specific_scenario:
                specific_train_losses.append(np.mean(train_losses))
                specific_test_losses.append(np.mean(test_losses))

        alpha_dir = os.path.join("results", f"alpha_known_{alpha_known}_unknown_{alpha_unknown}")
        os.makedirs(alpha_dir, exist_ok=True)
        save_results(results, filename=os.path.join(alpha_dir, "training_results.json"))
        plot_results(results, save_dir=alpha_dir)

        if case == specific_scenario:
            analyze_regularization_effect(model, alpha_unknown, alpha_dir, feature_names)

    scenario_name = f"{specific_scenario['model_name']}_With_Shortcut"
    plot_loss_vs_alpha(specific_train_losses, specific_test_losses, [a[0] for a in alphas], save_dir="results", scenario_name=scenario_name)

if __name__ == "__main__":
    main()

