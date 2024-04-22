# CCM Model Training Repository

This repository contains scripts for training and evaluating CBM and CCM models under distribution shift and shortcut learning using PyTorch.

## Scripts

1. `train.py`: This script is used for training CBM (Concept Bottleneck Model) and CCM (Concept Credible Model), with different configurations.

2. `eval.py`: This script provides functions for loading, plotting, and saving training results obtained from the `train.py` script.

3. `regularisation.py`: This script contains a function for calculating the EYE regularization term used in training the models.

4. `utils.py`: This script contains utility functions for preparing datasets, generating synthetic data, and creating custom PyTorch datasets and data loaders.

5. `models.py`: This script defines the architecture of the neural network models used in the conceptual models, including MLP (Multi-Layer Perceptron), CBM, and CCM.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/ioannapapanikolaou/CCM-DistributionShift.git
cd CCM-DistributionShift

