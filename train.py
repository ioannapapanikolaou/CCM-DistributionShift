import torch
from torch import optim, nn
from models import CCM, CBM, MLP, EYERegularization, CCMWithEYE
from utils import prepare_dataloaders, generate_dataset
from regularisation import EYE, cbm_loss
from eval import add_results, plot_results, save_results
from itertools import chain
import os
from cycler import cycler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models(device):
    input_size = 3  # input size based on dataset features
    concept_size = 5
    output_size = 1
    hidden_dim = 20

    # Separate instances for CBM and CCM
    net_c_cbm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_c_ccm = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_u = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_y = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)
    combined_net_y = MLP(input_dim=2 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)

    cbm = CBM(net_c=net_c_cbm, net_y=net_y)
    ccm = CCM(net_c=net_c_ccm, net_u=net_u, net_y=combined_net_y)

    # Combine parameters from both models without duplicates
    optimizer = optim.Adam(chain(cbm.parameters(), ccm.parameters()), lr=0.001)
    
    return cbm, ccm, optimizer

class CBMLoss(nn.Module):
    def __init__(self, lambda_concept=1):
        super(CBMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_concept = lambda_concept

    def forward(self, preds, targets):
        # preds and targets are tuples
        concepts_pred, y_pred = preds  # Unpack predictions
        concepts_label, y_label = targets  # Unpack targets assuming targets come as a tuple already

        # Ensure y_label is [batch_size, 1] to match y_pred
        y_label = y_label.unsqueeze(1) if y_label.dim() == 1 else y_label

        # Calculate loss
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm, ccm, optimizer = initialize_models(device)
    batch_size = 32

    alphas = [0.001]  

    cases = [
        {"model": cbm, "include_shortcut": False, "model_name": "CBM", "use_eye": False},
        {"model": cbm, "include_shortcut": True, "model_name": "CBM", "use_eye": False},
        {"model": ccm, "include_shortcut": False, "model_name": "CCM", "use_eye": True},
        {"model": ccm, "include_shortcut": True, "model_name": "CCM", "use_eye": True}
    ]

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
            features, known_concepts, target = generate_dataset(num_samples=1000, size_mean=10, include_shortcut=case['include_shortcut'], test_set=False)
            dataloader = prepare_dataloaders(features, known_concepts, target, batch_size=batch_size)
            losses = train(model, dataloader, optimizer, alpha, epochs=10, device=device, EYE_penalty=use_eye)

            if losses:
                add_results(results, case_label, losses)

        alpha_dir = os.path.join("results", f"alpha_{alpha}")
        os.makedirs(alpha_dir, exist_ok=True)
        save_results(results, filename=os.path.join(alpha_dir, "training_results.json"))
        plot_results(results, save_dir=alpha_dir)

if __name__ == "__main__":
    main()
