# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import os 


# def eye_regularization(beta1, beta2, lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut):
#     """
#     Calculate the EYE regularization term for given beta values.
#     Args:
#         beta1, beta2: numpy arrays of parameter values.
#         lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut: regularization coefficients.
#     Returns:
#         numpy array of the regularization term values.
#     """
#     term1 = np.abs(beta1)
#     term2 = np.sqrt(beta1**2 + beta2**2)
#     term3 = beta2**2

#     eye_reg = lambda_eye_known * term1 + lambda_eye_unknown * term2 + lambda_eye_shortcut * term3
#     return eye_reg

# def generate_contour_plot(lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut, save_dir):
#     """Generate and save contour plots for the EYE regularization term."""
#     beta1 = np.linspace(-1.5, 1.5, 400)
#     beta2 = np.linspace(-1.5, 1.5, 400)
#     B1, B2 = np.meshgrid(beta1, beta2)

#     eye_reg_values = eye_regularization(B1, B2, lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut)

#     plt.figure(figsize=(8, 6))
#     CS = plt.contour(B1, B2, eye_reg_values, levels=10, colors='black')
#     plt.clabel(CS, inline=1, fontsize=10)
    
#     plt.xlabel(r'$\beta_1$')
#     plt.ylabel(r'$\beta_2$')
#     plt.title(f'EYE Regularization Contour Plot')
#     plt.grid(True)

#     if save_dir is not None:
#         plt.savefig(os.path.join(save_dir, 'eye_regularization_contour_plot.png'))
#     else:
#         plt.show()

# # Example usage
# generate_contour_plot(lambda_eye_known=1.0, lambda_eye_unknown=1.0, lambda_eye_shortcut=1.0, save_dir="results/contour_plots")
# generate_contour_plot(lambda_eye_known=1.0, lambda_eye_unknown=10.0, lambda_eye_shortcut=1.0, save_dir="results/contour_plots")
# generate_contour_plot(lambda_eye_known=1.0, lambda_eye_unknown=1.0, lambda_eye_shortcut=10.0, save_dir="results/contour_plots")

import numpy as np
import matplotlib.pyplot as plt
import torch
import os 

def eye_regularization(beta1, beta2, lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut):
    """
    Calculate the EYE regularization term for given beta values.
    Args:
        beta1, beta2: numpy arrays of parameter values.
        lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut: regularization coefficients.
    Returns:
        numpy array of the regularization term values.
    """
    term1 = np.abs(beta1)
    term2 = np.sqrt(beta1**2 + beta2**2)
    term3 = beta2**2

    eye_reg = lambda_eye_known * term1 + lambda_eye_unknown * term2 + lambda_eye_shortcut * term3
    return eye_reg

def generate_contour_plot(lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut, save_dir):
    """Generate and save contour plots for the EYE regularization term."""
    beta1 = np.linspace(-1.5, 1.5, 400)
    beta2 = np.linspace(-1.5, 1.5, 400)
    B1, B2 = np.meshgrid(beta1, beta2)

    eye_reg_values = eye_regularization(B1, B2, lambda_eye_known, lambda_eye_unknown, lambda_eye_shortcut)

    plt.figure(figsize=(8, 6))
    CS = plt.contour(B1, B2, eye_reg_values, levels=10, colors='black')
    plt.clabel(CS, inline=1, fontsize=10)
    
    plt.xlabel(r'$\beta_1$')
    plt.ylabel(r'$\beta_2$')
    plt.title(f'EYE Regularization Contour Plot')
    plt.grid(True)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'eye_regularization_contour_plot.png'))
    else:
        plt.show()
