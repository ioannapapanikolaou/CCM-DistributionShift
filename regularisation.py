import torch
import torch.nn as nn

class cbm_loss(nn.Module):
    def __init__(self, lambda_concept=1):
        super(cbm_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_concept = lambda_concept

    def forward(self, y_pred, y_label):
        concepts_pred, y_pred = y_pred
        concepts_label, y_label = y_label[:, :-1], y_label[:, -1]
        return self.mse_loss(y_pred, y_label) + self.lambda_concept * self.mse_loss(concepts_pred, concepts_label)


# for contour plot
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
        l2_norm_x = torch.norm(theta_x, 2)**2
        l2_norm_c = torch.norm(theta_c, 2)**2
        l2_norm_s = torch.norm(theta_s, 2)**2
        
        return self.lambda_eye_known * l2_norm_c + self.lambda_eye_unknown * (l1_norm_x + torch.sqrt(l2_norm_x)) + self.lambda_eye_shortcut * l2_norm_s
