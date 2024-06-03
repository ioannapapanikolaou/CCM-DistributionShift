import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import os
import matplotlib.pyplot as plt
from cycler import cycler
from eval import add_results, plot_results, save_results

# class MLP(nn.Module):
#     """ 
#     A simple Multi-Layer Perceptron (MLP) model.
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.layers(x)

# class CBM(nn.Module):
#     """
#     Concept Bottleneck Model (CBM) that uses a concept network and a task network.
#     """
#     def __init__(self, net_c, net_y):
#         super(CBM, self).__init__()
#         self.net_c = net_c
#         self.net_y = net_y

#     def forward(self, x):
#         concepts = self.net_c(x)
#         y_pred = self.net_y(concepts)
#         return concepts, y_pred

# class CCM(nn.Module):
#     """
#     Concept Credible Model (CCM) that uses separate networks for known and unknown concepts.
#     """
#     def __init__(self, net_c, net_u, net_y):
#         super(CCM, self).__init__()
#         self.net_c = net_c
#         self.net_u = net_u
#         self.net_y = net_y

#     def forward(self, x):
#         o_c = self.net_c(x)
#         o_u = self.net_u(x)
#         combined_output = torch.cat((o_c, o_u), dim=1)
#         y_pred = self.net_y(combined_output)
#         return o_c, y_pred  # Make sure to return exactly two values
    
    


# Method 1: regularise as in paper 
# class EYERegularization(nn.Module):
#     def __init__(self, lambda_eye=1.0):
#         super(EYERegularization, self).__init__()
#         self.lambda_eye = lambda_eye

#     def forward(self, theta_x, theta_c):
#         theta_x = torch.cat([param.view(-1) for param in theta_x])
#         theta_c = torch.cat([param.view(-1) for param in theta_c])
        
#         l1_norm = torch.norm(theta_x, 1)
#         l2_norm = torch.norm(theta_x, 2)**2
#         l2_norm_c = torch.norm(theta_c, 2)**2
#         eye_reg = l1_norm + torch.sqrt(l2_norm) + l2_norm_c
#         return self.lambda_eye * eye_reg


# # Define the CCM model incorporating EYE regularization
# class CCMWithEYE(nn.Module):
#     def __init__(self, net_c, net_u, net_y, lambda_eye):
#         super(CCMWithEYE, self).__init__()
#         self.net_c = net_c
#         self.net_u = net_u
#         self.net_y = net_y
#         self.eye_regularization = EYERegularization(lambda_eye=lambda_eye)

#     def forward(self, x):
#         c = self.net_c(x)
#         u = self.net_u(x)
#         cu = torch.cat((c, u), dim=1)
#         y = self.net_y(cu)
#         return c, y


# # Method 2 functions with regularisation at different terms. 
# class EYERegularization(nn.Module):
#     def __init__(self, lambda_eye_known=1.0, lambda_eye_unknown=1.0):
#         super(EYERegularization, self).__init__()
#         self.lambda_eye_known = lambda_eye_known
#         self.lambda_eye_unknown = lambda_eye_unknown

#     def forward(self, theta_x, theta_c):
#         # Apply different regularization strengths to known and unknown features
#         theta_x_known = theta_x[0:1]  # Adjust these indices based on the known features in theta_x
#         theta_x_unknown = theta_x[1:]  # Adjust these indices based on the unknown features in theta_x

#         theta_x_known = torch.cat([param.view(-1) for param in theta_x_known])
#         theta_x_unknown = torch.cat([param.view(-1) for param in theta_x_unknown])
        
#         l1_norm_known = torch.norm(theta_x_known, 1)
#         l2_norm_known = torch.norm(theta_x_known, 2)**2

#         l1_norm_unknown = torch.norm(theta_x_unknown, 1)
#         l2_norm_unknown = torch.norm(theta_x_unknown, 2)**2

#         eye_reg = self.lambda_eye_known * (l1_norm_known + torch.sqrt(l2_norm_known)) + \
#                   self.lambda_eye_unknown * (l1_norm_unknown + torch.sqrt(l2_norm_unknown))
#         return eye_reg


# class CCMWithEYE(nn.Module):
#     def __init__(self, net_c, net_u, net_y, lambda_eye_known, lambda_eye_unknown):
#         super(CCMWithEYE, self).__init__()
#         self.net_c = net_c
#         self.net_u = net_u
#         self.net_y = net_y
#         self.eye_regularization = EYERegularization(lambda_eye_known=lambda_eye_known, lambda_eye_unknown=lambda_eye_unknown)

#     def forward(self, x):
#         c = self.net_c(x)
#         u = self.net_u(x)
#         cu = torch.cat((c, u), dim=1)
#         y = self.net_y(cu)
#         return c, y

# # Method 3: Explicitly handle shortcuts.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from regularisation import EYERegularization

# class MLP(nn.Module):
#     """ 
#     A simple Multi-Layer Perceptron (MLP) model.
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.layers(x)

# class CBM(nn.Module):
#     """
#     Concept Bottleneck Model (CBM) that uses a concept network and a task network.
#     """
#     def __init__(self, net_c, net_y):
#         super(CBM, self).__init__()
#         self.net_c = net_c
#         self.net_y = net_y

#     def forward(self, x):
#         concepts = self.net_c(x)
#         y_pred = self.net_y(concepts)
#         return concepts, y_pred

# class CCM(nn.Module):
#     """
#     Concept Credible Model (CCM) that uses separate networks for known and unknown concepts.
#     """
#     def __init__(self, net_c, net_u, net_y):
#         super(CCM, self).__init__()
#         self.net_c = net_c
#         self.net_u = net_u
#         self.net_y = net_y

#     def forward(self, x):
#         o_c = self.net_c(x)
#         o_u = self.net_u(x)
#         combined_output = torch.cat((o_c, o_u), dim=1)
#         y_pred = self.net_y(combined_output)
#         return o_c, y_pred  # Make sure to return exactly two values

# # Define the CCM model incorporating EYE regularization
# class CCMWithEYE(nn.Module):
#     def __init__(self, net_c, net_u, net_y, net_s=None, lambda_eye_known=1.0, lambda_eye_unknown=1.0, lambda_eye_shortcut=1.0):
#         super(CCMWithEYE, self).__init__()
#         self.net_c = net_c
#         self.net_u = net_u
#         self.net_s = net_s
#         self.net_y = net_y
#         self.eye_regularization = EYERegularization(lambda_eye_known=lambda_eye_known, lambda_eye_unknown=lambda_eye_unknown, lambda_eye_shortcut=lambda_eye_shortcut)

#     def forward(self, x):
#         c = self.net_c(x)
#         u = self.net_u(x)
#         s = self.net_s(x) if self.net_s else torch.zeros_like(c)
#         cu = torch.cat((c, u, s), dim=1)
#         y = self.net_y(cu)
#         return c, y



# Method 3.5: Explicitly handle shortcuts. Completeness score.
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
        y_pred = self.net_y(concepts)
        return concepts, y_pred

class CCM(nn.Module):
    """
    Concept Credible Model (CCM) that uses separate networks for known and unknown concepts.
    """
    def __init__(self, net_c, net_u, net_y):
        super(CCM, self).__init__()
        self.net_c = net_c
        self.net_u = net_u
        self.net_y = net_y

    def forward(self, x):
        o_c = self.net_c(x)
        o_u = self.net_u(x)
        combined_output = torch.cat((o_c, o_u), dim=1)
        y_pred = self.net_y(combined_output)
        return o_c, y_pred  # Make sure to return exactly two values

class EYERegularization(nn.Module):
    def __init__(self, lambda_eye_known=1.0, lambda_eye_unknown=1.0, lambda_eye_shortcut=1.0):
        super(EYERegularization, self).__init__()
        self.lambda_eye_known = lambda_eye_known
        self.lambda_eye_unknown = lambda_eye_unknown
        self.lambda_eye_shortcut = lambda_eye_shortcut

    def forward(self, theta_x, theta_c, theta_s):
        theta_x = torch.cat([param.view(-1) for param in theta_x])
        theta_c = torch.cat([param.view(-1) for param in theta_c])
        theta_s = torch.cat([param.view(-1) for param in theta_s])
        
        l1_norm_x = torch.norm(theta_x, 1)
        l1_norm_s = torch.norm(theta_s, 1)
        l2_norm_c = torch.norm(theta_c, 2)**2
        eye_reg = l1_norm_x + l1_norm_s + torch.sqrt(l2_norm_c)
        return self.lambda_eye_known * eye_reg

# Define the CCM model incorporating EYE regularization
class CCMWithEYE(nn.Module):
    def __init__(self, net_c, net_u, net_y, net_s, lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut):
        super(CCMWithEYE, self).__init__()
        self.net_c = net_c
        self.net_u = net_u
        self.net_y = net_y
        self.net_s = net_s
        self.eye_regularization = EYERegularization(lambda_eye_known=lambda_eye_known, lambda_eye_unknown=lambda_eye_unknown, lambda_eye_shortcut=lambda_eye_shortcut)

    def forward(self, x):
        c = self.net_c(x)
        u = self.net_u(x)
        s = self.net_s(x)
        cus = torch.cat((c, u, s), dim=1)
        y = self.net_y(cus)
        return c, y


