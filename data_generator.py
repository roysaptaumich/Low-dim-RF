import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
import torch
import torch.distributions as dist

def generate_data(type, n_samples, INPUT_DIM, RANK, CENTER, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # #####################################
    # # Generate Two-moons dataset
    # #####################################
    if type == 'two_moons':
        D1_2D_np, labels_np = make_moons(
            n_samples=n_samples, 
            noise=0.05,  # Add a small amount of Gaussian noise
            random_state=42
        )
        SEPARATION_DISTANCE = -0.9

        # 2. Convert to PyTorch Tensor and ensure correct device/dtype
        D1_2D = 3 * torch.tensor(D1_2D_np, dtype=torch.float32, device=device)
        labels = torch.tensor(labels_np, dtype=torch.bool, device=device)

        moon0_mask = (labels == 0)
        moon1_mask = (labels == 1)

        # Shift Moon 0 (e.g., left)
        D1_2D[moon0_mask, 1] -= SEPARATION_DISTANCE 

        # Shift Moon 1 (e.g., right)
        D1_2D[moon1_mask, 1] += SEPARATION_DISTANCE
        # 3. Embed the 2D data into the INPUT_DIM space (1000D)
        # We place the 2D data into the first two dimensions (dim 0 and 1)
        # and pad the rest with zeros: (n_samples, INPUT_DIM - 2)
        padding = torch.zeros((n_samples, INPUT_DIM - 2), device=device)

        # Concatenate the 2D data with the padding
        D1_embedded = torch.cat((D1_2D, padding), dim=1)

        # 4. Apply the centering/shift (same as your original code)
        D1 = D1_embedded + CENTER

    ####################################
    # Generate Swiss roll
    ####################################
    if type == 'swiss_roll':

        # Swiss roll in R^3
        X_np, _ = make_swiss_roll(
            n_samples=n_samples,
            noise=0.05,
            random_state=42,
        )

        # Standardize
        X_np = X_np - X_np.mean(axis=0)
        X_np = X_np / X_np.std()

        X = torch.tensor(X_np, dtype=torch.float32, device=device)

        # Scale similarly to the moons experiment
        X = 3 * X

        # Embed into ambient dimension
        padding = torch.zeros((n_samples, INPUT_DIM - 3), device=device)
        D1 = torch.cat((X, padding), dim=1)

        # Random rotation (recommended)
        G = torch.randn(INPUT_DIM, INPUT_DIM, device=device)
        Q, _ = torch.linalg.qr(G)
        D1 = D1 @ Q

        # Global shift
        D1 = D1 + CENTER
    ####################################
    # Generate Gaussian with low-rank covariance
    # first sample RANK-dim Gaussian
    ####################################
    if type == 'low_rank_Gaussian':
        pi_1_rank = dist.MultivariateNormal(torch.zeros(RANK, device=device), torch.eye(RANK, device=device))
        D1_rank = pi_1_rank.sample([n_samples])
        # pad INPUT_DIM - RANK zeros to each sample
        padding = torch.zeros((n_samples, INPUT_DIM - RANK),device=device)
        D1 = torch.cat((D1_rank, padding), dim=1) + CENTER

    return D1