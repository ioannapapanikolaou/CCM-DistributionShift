import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ 
    A simple Multi-Layer Perceptron (MLP) model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class CBM(nn.Module):
    """
    Concept Bottleneck Model (CBM) that uses a concept network and a task network.
    """
    def __init__(self, net_c, net_y):
        super(CBM, self).__init__()
        self.net_c = net_c
        self.net_y = net_y

    def forward(self, x):
        concepts = self.net_c(x)
        return self.net_y(concepts)

class CCM(nn.Module):
    """
    Concept Credible Model (CCM) that uses separate networks for known and unknown concepts.
    """
    def __init__(self, net_c, net_u, net_y, c_no_grad=True, u_no_grad=False):
        super(CCM, self).__init__()
        self.net_c = net_c
        self.net_u = net_u
        self.net_y = net_y
        if c_no_grad:
            self.freeze_params(self.net_c)
        if u_no_grad:
            self.freeze_params(self.net_u)

    def freeze_params(self, net):
        for param in net.parameters():
            param.requires_grad = False

    def forward(self, x):
        o_c = self.net_c(x)
        o_u = self.net_u(x)
        combined_output = torch.cat((o_c, o_u), dim=1)
        return self.net_y(combined_output)
