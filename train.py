import torch
from torch import optim
from models import CCM, CBM, MLP
from utils import prepare_dataloaders, generate_dataset
from regularisation import EYE
from eval import add_results, plot_results, save_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models(device):
    # Define model dimensions
    input_size = 4
    concept_size = 5
    output_size = 1
    hidden_dim = 20

    # Initialize model components for both CBM and CCM
    net_c = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_u = MLP(input_dim=input_size, hidden_dim=hidden_dim, output_dim=concept_size).to(device)
    net_y = MLP(input_dim=concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)  # Used for CBM
    combined_net_y = MLP(input_dim=2 * concept_size, hidden_dim=hidden_dim, output_dim=output_size).to(device)  # Used for CCM

    cbm = CBM(net_c=net_c, net_y=net_y)
    ccm = CCM(net_c=net_c, net_u=net_u, net_y=combined_net_y)
    return cbm, ccm, optim.Adam(list(cbm.parameters()) + list(ccm.parameters()), lr=0.001)

def train(model, dataloader, optimizer, epochs=10):
    model.train()
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze()

            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        losses.append(total_loss / len(dataloader))
    return losses

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm, ccm, optimizer = initialize_models(device)
    all_params = list(cbm.parameters()) + list(ccm.parameters())
    unique_params = set(all_params)
    if len(all_params) != len(unique_params):
        print("Duplicate parameters found!")

    batch_size = 32
    results = {}

    cases = [
        {"model": cbm, "include_shortcut": False, "model_name": "CBM"},
        {"model": cbm, "include_shortcut": True, "model_name": "CBM"},
        {"model": ccm, "include_shortcut": False, "model_name": "CCM"},
        {"model": ccm, "include_shortcut": True, "model_name": "CCM"}
    ]

    for case in cases:
        print(f"Training case: {case['model_name']} {'with' if case['include_shortcut'] else 'without'} shortcut")
        features, labels = generate_dataset(1000, 10, include_shortcut=case['include_shortcut'])
        dataloader = prepare_dataloaders(features, labels, batch_size)
        losses = train(case['model'], dataloader, optimizer, epochs=10)

        case_label = f"{case['model_name']} {'With' if case['include_shortcut'] else 'Without'} Shortcut"
        add_results(results, case_label, losses, 10)
        print(f"Training loss for the last epoch: {losses[-1]}")

    # Save results to a file
    # save_results(results, filename="training_results.json")
    save_results(results, filename="results/training_results.json")
    # Load and plot results 
    results_directory = "results"
    plot_results(results, save_dir=results_directory)

if __name__ == "__main__":
    main()
