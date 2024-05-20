import torch
from torch import optim, nn
from models import CCM, CBM, MLP, EYERegularization, CCMWithEYE
from utils import prepare_dataloaders, generate_dataset, compare_distributions
from regularisation import EYE, cbm_loss
from eval import add_results, plot_results, save_results, plot_loss_vs_alpha  # Import the function
from itertools import chain
import os
import numpy as np

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

def train(model, dataloader, optimizer, alpha, epochs=10, device=None, EYE_penalty=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.train()
    loss_func = CBMLoss(lambda_concept=1).to(device)
    eye_regularization = EYERegularization(lambda_eye=alpha).to(device)
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
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
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        losses.append(average_loss)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')

    return losses

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm, ccm, optimizer = initialize_models(device)
    batch_size = 32

    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    overall_train_losses = []
    overall_test_losses = []

    cases = [
        {"model": cbm, "include_shortcut": False, "model_name": "CBM", "use_eye": False},
        {"model": cbm, "include_shortcut": True, "model_name": "CBM", "use_eye": False},
        {"model": ccm, "include_shortcut": False, "model_name": "CCM", "use_eye": True},
        {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
    ]

    # Specify the scenario for loss vs alpha plot
    specific_scenario = {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
    specific_train_losses = []
    specific_test_losses = []

    for alpha in alphas:
        results = {}
        for case in cases:
            model = case["model"]
            model_name = case["model_name"]
            use_eye = case["use_eye"]

            if use_eye:
                model = CCMWithEYE(model.net_c, model.net_u, model.net_y, lambda_eye=alpha)
                optimizer = optim.Adam(chain(model.net_c.parameters(), model.net_u.parameters(), model.net_y.parameters()), lr=0.001)

            case_label = f"{model_name} {'With' if case['include_shortcut'] else 'Without'} Shortcut, Alpha={alpha}"
            print(f"Training case: {case_label}")
            train_features, train_known_concepts, train_target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=False)
            train_dataloader = prepare_dataloaders(train_features, train_known_concepts, train_target, batch_size=batch_size)

            test_features, test_known_concepts, test_target = generate_dataset(num_samples=200, size_mean=10, include_shortcut=case['include_shortcut'], test_set=True)
            test_dataloader = prepare_dataloaders(test_features, test_known_concepts, test_target, batch_size=batch_size)

            # Directory to save distribution plots
            plot_save_dir = os.path.join("results", f"alpha_{alpha}", "distributions")
            
            # Compare feature distributions and save plots
            compare_distributions(train_features, test_features, alpha, plot_save_dir)

            # Train the model
            train_losses = train(model, train_dataloader, optimizer, alpha, epochs=1000, device=device, EYE_penalty=use_eye)

            # Evaluate the model on the test set
            test_loss = evaluate(model, test_dataloader, CBMLoss(lambda_concept=1).to(device), device)

            if train_losses:
                add_results(results, case_label + " - Train", train_losses)
                add_results(results, case_label + " - Test", [test_loss] * len(train_losses))

            # Collect losses for the specific scenario
            if case == specific_scenario:
                specific_train_losses.append(np.mean(train_losses))
                specific_test_losses.append(test_loss)

        alpha_dir = os.path.join("results", f"alpha_{alpha}")
        os.makedirs(alpha_dir, exist_ok=True)
        save_results(results, filename=os.path.join(alpha_dir, "training_results.json"))
        plot_results(results, save_dir=alpha_dir)

    # Plot loss vs alpha for the specific scenario
    scenario_name = f"{specific_scenario['model_name']}_With_Shortcut"
    plot_loss_vs_alpha(specific_train_losses, specific_test_losses, alphas, save_dir="results", scenario_name=scenario_name)

if __name__ == "__main__":
    main()
