import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import os
import matplotlib.pyplot as plt
from cycler import cycler
from eval import add_results, plot_results, save_results
from regularisation import EYERegularization

class MLP(nn.Module):
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
    def __init__(self, net_c, net_y):
        super(CBM, self).__init__()
        self.net_c = net_c
        self.net_y = net_y

    def forward(self, x):
        concepts = self.net_c(x)
        y_pred = self.net_y(concepts)
        return concepts, y_pred

class CCM(nn.Module):
    def __init__(self, known_concept_dim, unknown_concept_dim, hidden_dim, output_dim, alpha):
        super(CCM, self).__init__()
        self.net_c = MLP(input_dim=known_concept_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.net_u = MLP(input_dim=unknown_concept_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.net_y = MLP(input_dim=hidden_dim*2, hidden_dim=hidden_dim, output_dim=output_dim)
        self.eye_regularization = EYERegularization(lambda_eye_known=alpha, lambda_eye_unknown=alpha, lambda_eye_shortcut=alpha)

    def forward(self, x):
        known_concepts = x[:, :2]  # Assuming the first 2 features are known concepts
        unknown_concepts = x[:, 2:]  # The rest are unknown concepts
        
        c = self.net_c(known_concepts)
        u = self.net_u(unknown_concepts)
        cu = torch.cat((c, u), dim=1)
        y = self.net_y(cu)
        return c, y

    def apply_eye_regularization(self, alpha):
        # Update EYE regularization parameters
        self.eye_regularization.lambda_eye_known = alpha
        self.eye_regularization.lambda_eye_unknown = alpha
        self.eye_regularization.lambda_eye_shortcut = alpha
        # Apply EYE regularization (this should be integrated into the training loop)

        # Assuming theta_x, theta_c, and theta_s are parameters from respective networks
        theta_x = [param for param in self.net_y.parameters()]
        theta_c = [param for param in self.net_c.parameters()]
        theta_s = [param for param in self.net_u.parameters()]
        return self.eye_regularization(theta_x, theta_c, theta_s)
