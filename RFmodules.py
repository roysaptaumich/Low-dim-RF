import os 
import ot
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from scipy.stats import gaussian_kde

import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
     
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_p_hat_t(y, X, t):
    """
    log p̂_t(y) using Gaussian kernel density + log-sum-exp trick
    y: (M, d)
    X: (N, d)
    t: scalar
    
    returns: (M,)
    """
    if t == 1: raise ValueError("t should be less than 1 to avoid division by zero.")
    # M, d = y.shape
    # N = X.shape[0]
    X_t_mean = t * X  # (N, d)
    sigma_t  = 1-t

    # pairwise distances
    diff = y[:, None, :] - X_t_mean[None, :, :]  # (M, N, d)
    sq_dist = (diff ** 2).sum(dim=-1)  # (M, N)

    exponents = -sq_dist / (2 * sigma_t**2)
    lse = torch.logsumexp(exponents, dim=1)  # (M,)

   

    return lse  # (M,)

def score_kernel_exact(y, X, t):
    """
    Exact gradient of log p̂_t(y) wrt y
    y: (M, d)
    X: (N, d)
    t: scalar
    returns: (M, d)
    """
    sigma_t = 1 - t
    X_t_mean = t * X  # (N, d)
    
    # pairwise distances
    diff = y[:, None, :] - X_t_mean[None, :, :]  # (M, N, d)
    sq_dist = (diff ** 2).sum(dim=-1)  # (M, N)

    # unnormalized weights
    logits = -sq_dist / (2 * sigma_t**2)
    weights = torch.softmax(logits, dim=1)  # (M, N)

    # weighted sum of (y - t x_i)
    grad = - (weights[..., None] * diff).sum(dim=1) / (sigma_t**2)
    return grad * sigma_t**2  # (M, d)

# def log_p_hat_t_vectorized(y, X, t_vec):
#     """
#     log p̂_t(y) using Gaussian kernel density + log-sum-exp trick
#     y: (Total, d)
#     X: (N, d)
#     t_vec: (Total,) where Total = T * M * N
    
#     returns: (Total,)
#     """
#     Total, d = y.shape
#     N = X.shape[0]

#     # Reshape t_vec to broadcast: (Total, 1, 1)
#     t_vec_exp = t_vec[:, None, None]
    
#     # X_t_mean: (Total, N, d)
#     # The mean of the kernel t * X is now unique for each y based on its t value
#     X_t_mean = t_vec_exp * X[None, :, :]  # (Total, N, d)

#     # sigma_t: (Total, 1)
#     sigma_t = 1 - t_vec_exp[:, :, 0] # (Total, 1)
    
#     # pairwise distances
#     diff = y[:, None, :] - X_t_mean  # (Total, N, d)
#     sq_dist = (diff ** 2).sum(dim=-1)  # (Total, N)

#     # exponents: -sq_dist / (2 * sigma_t^2)
#     exponents = -sq_dist / (2 * sigma_t**2)
#     lse = torch.logsumexp(exponents, dim=1)  # (Total,)

#     # The normalization constant (log(1/N)) is often omitted as it doesn't affect the gradient (score)
#     # The 1/N term should be exponents + log(1/N), but logsumexp handles the constant up to an additive shift.
    
#     return lse  # (Total,)

#@torch.no_grad()
def score_blurred(xt, X, t, sigma_noise = 0.1, device = device):
    """
    xt : (T+1) x N x d (time, training_sample, dimension) = the tensor of (1-t) * z + t * x
    
    X: N x d

    t : (T+1 - dim array) time sequence 

    device : available device

    sigma_noise (scalar) : noise variacne for blurring

    Output : T x N x d : avoids t = 1
    """
    _, N , d = xt.shape 
    M = 100
    noise = sigma_noise * torch.randn(M, d, device = device) # M x d

    score_blurred = []
    for i in range(len(t)):
        if t[i].item() <1:
            time = t[i].item()
            x_noised  = xt[i][None, :, :] + noise[ :, None, :] #  M x N x d   

            y = x_noised.reshape( M * N, d) # (M * N) x d
            #y = y.clone().detach().requires_grad_(True)
            # compute the log p_hat(y) for all y
            #logp = log_p_hat_t(y, X ,time) # ( M * N, )
            
            # print("y.requires_grad:", y.requires_grad)
            # print("logp.requires_grad:", logp.requires_grad)
            # print("logp.grad_fn:", logp.grad_fn)
            # Compute gradient wrt y
            # grad = torch.autograd.grad(logp.sum(), y, retain_graph=False,
            #                            create_graph=False)[0]  # ( M * N, d)
            #print('iteration:', i, 'grad.requires_grad:', grad.requires_grad)
            grad = score_kernel_exact(y, X, time) # ( M * N, d)
            # Reshape back to (T, n_samples, d)
            grad = grad.view(M, N, d)

            # Monte Carlo average across samples
            s_k = grad.mean(dim=0)  # ( N, d)
            score_blurred.append(s_k)
    
    return torch.stack(score_blurred) # (T, N, d)


# def score_blurred_parallel(xt, X, t, sigma_noise=0.1, device=None):
#     """
#     Computes the blurred score function for all time steps t simultaneously.

#     xt : (T, N, d) - Time-dependent training samples (no t=1 needed)
#     X: (N, d) - Original training data
#     t : (T,) - Time sequence (must be < 1)
    
#     Output : (T, N, d) 
#     """
#     T, N, d = xt.shape 
#     M = 100  # Number of Monte Carlo samples

#     # 1. Check for time t=1 and move tensors to device
#     if torch.any(t >= 1.0):
#         raise ValueError("All time steps t must be less than 1 to avoid division by zero.")
    
#     if device is None:
#         device = xt.device

#     # --- Monte Carlo Sampling (Batched) ---
#     # Create M * T * N total samples for the current batch
    
#     # 2. Generate T * M * d noise samples
#     # Reshape: (T, M, 1, d)
#     # The noise is shared across all N training samples (xt[i]) but is unique per time step and MC sample
#     noise = sigma_noise * torch.randn(T, M, 1, d, device=device) 

#     # 3. Create the input 'y' for log_p_hat_t
#     # Expand xt: (T, 1, N, d) + noise: (T, M, 1, d) -> (T, M, N, d)
#     x_noised = xt[:, None, :, :] + noise
    
#     # Reshape to a single large batch for log_p_hat_t: (T * M * N) x d
#     y = x_noised.reshape(T * M * N, d)
    
#     # --- Compute Log p_hat(y) and Gradient (Score) ---
    
#     # 4. Enable gradient tracking for y
#     y = y.clone().detach().requires_grad_(True)
    
#     # log_p_hat_t requires a single time scalar t. 
#     # To parallelize across T, we need a slight modification (see notes below)
#     # or loop over T for the logp calculation (which we avoid here by passing T-sized time)
#     # Assuming log_p_hat_t is refactored to handle a T-sized time tensor:
    
#     # log_p_hat_t_vectorized needs the time t replicated: (T * M * N,)
#     t_expanded = t[:, None, None].expand(T, M, N).reshape(T * M * N)
    
#     # 5. Compute the log-probability for the entire batch
#     # Requires a modified log_p_hat_t (see Note 1)
#     logp_full = log_p_hat_t_vectorized(y, X, t_expanded) # (T * M * N,)

#     # 6. Compute gradient wrt y
#     # Sum the log probabilities to get the total gradient
#     grad = torch.autograd.grad(logp_full.sum(), y, retain_graph=False,
#                                    create_graph=False)[0]  # (T * M * N, d)

#     # 7. Reshape and Monte Carlo Average
#     # Reshape back: (T, M, N, d)
#     grad = grad.view(T, M, N, d)

#     # Average across the M Monte Carlo samples (dim=1)
#     s_k = grad.mean(dim=1)  # (T, N, d)
    
#     return s_k # (T, N, d)

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t): #does not need time for now
        inputs = torch.cat([x_input, t], dim=-1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x




class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=64, scale=10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        # x: (..., in_dim)
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNetFourier(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim=256, num_frequencies=64):
        super().__init__()
        self.fourier = FourierFeatures(input_dim + 1, num_frequencies)  # t concatenated with x
        in_dim = 2 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        # x: (N, d), t: (N, 1)
        inp = torch.cat([x, t], dim=-1)
        ff = self.fourier(inp)
        return self.net(ff)




class VelocityNet(nn.Module):
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.fc_in  = nn.Linear(input_dim + 1, h_dim)
        self.fc2    = nn.Linear(h_dim, h_dim)
        self.fc3    = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)
    
    def forward(self, x, t, act=F.gelu):
        t =  t.squeeze().view(t.shape[0], -1)  # Ensure t has the correct dimensions
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)

class VelocityNet2(nn.Module):
    def __init__(self, input_dim, h_dim=64):
        super().__init__()
        self.fc_in  = nn.Linear(input_dim + 1, h_dim)
        self.fc2    = nn.Linear(h_dim, h_dim)
        self.fc3    = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, input_dim)
    
    def forward(self, x, t, act=F.gelu):
        t =  t.squeeze().view(t.shape[0], -1)  # Ensure t has the correct dimensions
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = x/((1-t) + 1e-3)  # Scale the output by 1/(1-t) to enhance stability as t approaches 1
        return self.fc_out(x)

class MLPVelocity2(nn.Module):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None):
        super().__init__()
        output_dim = output_dim or dim
        self.mlp = MLP([dim + 1, *hidden_sizes, output_dim])

    def forward(self, x, t):
        t = t.squeeze().view(t.shape[0], -1)
        x = x.view(x.shape[0], -1)
        return self.mlp(torch.cat((x, t), dim=1))

# NON-linear gated velocity structure inspired by the gating mechanism in RNNs, which can help capture complex interactions between time and state
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Fourier Features Embedding
# -----------------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_frequencies=64, scale=10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        # Ensure x is 2D: (batch_size, in_dim)
        if x.ndim == 1:
            x = x.unsqueeze(1)  # (N,) -> (N,1)
        elif x.ndim > 2:
            x = x.view(x.shape[0], -1)
        proj = 2 * torch.pi * x @ self.B  # (batch_size, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# -----------------------------
# Nonlinear Block
# -----------------------------
class NonlinearBlock(nn.Module):
    def __init__(self, x_dim, t_dim, hidden_dim=256, num_freq=64):
        super().__init__()
        # Only embedding t here; x is fed directly
        self.t_embed = FourierFeatures(t_dim, num_freq)

        in_dim = x_dim + 2 * num_freq  # x + Fourier embedding of t

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t):
        # Ensure x is 2D
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        # Ensure t is 2D
        if t.ndim == 1:
            t = t.unsqueeze(1)

        ft = self.t_embed(t)  # (batch_size, 2*num_freq)
        h = torch.cat([x, ft], dim=-1)
        return self.net(h)


# -----------------------------
# Nonlinear Gated Velocity
# -----------------------------
class NonlinearGatedVelocity(nn.Module):
    """
    v(x,t) = v_smooth(x,t) + g(t) * v_stiff(x,t)
    Fourier embedding for t; handles arbitrary batch size
    """
    def __init__(self, d, hidden_dim=512, num_freq=64):
        super().__init__()
        self.v_smooth = NonlinearBlock(d, t_dim=1, hidden_dim=hidden_dim, num_freq=num_freq)
        self.v_stiff  = NonlinearBlock(d, t_dim=1, hidden_dim=hidden_dim, num_freq=num_freq)

        # gating network uses only t
        self.t_embed = FourierFeatures(1, num_freq)
        self.gate = nn.Sequential(
            nn.Linear(2*num_freq, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        # Ensure x and t are 2D
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        if t.ndim == 1:
            t = t.unsqueeze(1)

        g = self.gate(self.t_embed(t))  # (batch_size, 1)
        v1 = self.v_smooth(x, t)
        v2 = self.v_stiff(x, t)
        return v1 + g * v2/(1-t + 1e-3)  # Scale stiff part by 1/(1-t) to enhance stability as t->1

#############
#############
class rectified_flow:
    def __init__(self, input_dim = 2, time_steps=2000, tiling = 'uniform',device = device):
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.device = device
        self.tiling = tiling
    def noise_images(self, x, t, z = None):
        """
        x : N x d, training data
        z : N x d, Guassian noise
        t : (T-dimensional array) time steps in [0,1] 
        """
        z = torch.randn_like(x) if z is None else z
        tt1 = (1-t)[:, None, None].to(self.device) # T x 1 x 1
        tt2 = t[:, None, None].to(self.device)
        return tt1 * z + tt2 * x, z  # T x N x d
    
    # Sampled timesteps will be used to train the model over those time points
    def sample_timesteps(self):
        T = self.time_steps
        if self.tiling == 'uniform':
            t = torch.linspace(0, 1, T+1).to(self.device)
            return t  # Ensure this is on the correct device
        if self.tiling == 'distorted':
            """
        Generate partition sequence t_j as in the given formula using PyTorch.
        
        Partition [0, 1/2]: 
            t0 = 0
            t1 = 1/n
            t_j = (1 + h) * t_{j-1}   for 2 ≤ j ≤ T/2
        
        Partition [1/2, 1]: 
            t_T = 1
            t_{T-1} = 1 - 1/T
            1 - t_j = (1 + h)(1 - t_{j+1})   for T/2 ≤ j ≤ T-2
        """
        t = torch.zeros(T + 1, device = self.device)
        h = np.power(T/2, 2/(T-2)) - 1.0
        # First partition [0, 1/2]
        t[0] = 0
        t[1] = 1.0 / T
        for j in range(2, T // 2 + 1):
            t[j] = (1 + h) * t[j - 1]

        # Second partition [1/2, 1]
        t[T] = 1.0
        t[T - 1] = 1.0 - 1.0 / T
        for j in range(T - 2, T // 2 - 1, -1):
            t[j] = 1.0 - (1 + h) * (1.0 - t[j + 1])

        return t

    def sample(self, model, time_seq, n = 2000):
        logging.info(f"Sampling {n} new data")
        traj = []
        model.eval()
        with torch.no_grad():
            xt = torch.randn((n, self.input_dim)).to(self.device)
            init_sample = xt.detach().clone()
            for i in tqdm(range(self.time_steps - 1)):
                t = (torch.ones(n, 1) * time_seq[i]).to(self.device)  # Ensure t is on the device
                xt += model(xt, t) * (time_seq[i+1] - time_seq[i])
                traj.append(xt.detach().clone())
        model.train()
        self.traj = torch.stack(traj)
        return xt, init_sample
    
    def sample_SB(self, model_SB, target_mean, time_seq, n = 2000):
        # Sampling using blurred scored

        logging.info(f"Sampling {n} new data")
        traj_SB = []
        model_SB.eval()
        with torch.no_grad():
            xt = torch.randn((n, self.input_dim)).to(self.device)
            init_sample = xt.detach().clone()
            for i in tqdm(range(self.time_steps - 1)):
                t = (torch.ones(n, 1) * time_seq[i]).to(self.device)  # Ensure t is on the device
                if time_seq[i] == 0: vt = target_mean - xt 
                # model_SB  = (1-t)^2 * actual_blurred_score. So, to get the velocity field we have
                # vt = x/t + 1/((1-t)*t) * model_SB(x) = x/t + (1-t)/t * actual_blurred_score
                # This enhances stability when 1-t is small
                else: vt = xt/time_seq[i] + 1/((1 - time_seq[i]) * time_seq[i]) *  model_SB(xt, t)
                
                xt += vt * (time_seq[i+1] - time_seq[i])
                traj_SB.append(xt.detach().clone())
        model_SB.train()
        self.traj_SB = torch.stack(traj_SB)
        return xt, init_sample

def train(args, device = device):
    print(args)
    #device = args['device']
    dataloader = args['dataloader']
    sample_size = args['sample_size']  # Keep it as an integer, no need to convert to a tensor
    time_steps = args['time_steps']

    model = MLP().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args['lr'])
    
    mse = nn.MSELoss()
    rect_flow = rectified_flow(input_dim = args['input_dim'], time_steps = time_steps, 
                               tiling = args['tiling'], device=device)
    logger = SummaryWriter(os.path.join("runs", args['run_name']))
    l = len(dataloader)
    loss_curve = []
    
    t = rect_flow.sample_timesteps()  # Timesteps on the same device

    # Mini-batch gradient descent
    for epoch in range(args['epochs']):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data, _) in enumerate(pbar):
            print('iteration:', i, 'epoch:', epoch)
            data = data.to(torch.float32)
            data = data.to(device)  # Ensure images are on the device
            
            
            x_t, z = rect_flow.noise_images(data, t)  # Ensure these tensors are on the correct device
            
            t_expanded = t.view(len(t), 1, 1).expand(len(t), data.shape[0], 1)
            print('x_t.shape, t_expanded.shape', x_t.shape, t_expanded.shape)
            # Usual velocity formulation
            predict_velocity = model(x_t, t_expanded)  # Model is on the device
            loss = mse(data - z, predict_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            


            
            

            loss_curve.append(loss.item()) # storing the training loss
            

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
           

    # Sample and return images
    sampled_data, init_samples = rect_flow.sample(model = model, time_seq = t, n=sample_size)
    
    return sampled_data, init_samples, loss_curve, rect_flow.traj
         
def rf_trainer(rectified_flow, D0, D1, label = 'loss', batch_size = 1024, num_epochs = 5000):
    model = rectified_flow.velocity_field
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = rectified_flow.get_loss

    losses = []

    #num_epochs = 5000
    for epoch in range(num_epochs):
        idx = torch.randint(0, D0.shape[0], (batch_size,))
        x0_batch = D0[idx]
        x1_batch = D1[idx]
        
        loss = criterion(x0_batch, x1_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], {label}: {loss.item():.4f}')
    return losses

def train_SB(args, device = device):
    print(args)
    #device = args['device']
    dataloader = args['dataloader']
    sample_size = args['sample_size']  # Keep it as an integer, no need to convert to a tensor
    sigma_blurring  = args['sigma_blurring']
    time_steps = args['time_steps']

    model_SB = ScoreNetFourier(input_dim = dataloader.dataset[0][0].shape[0]).to(device)

    optimizer_SB = optim.AdamW(model_SB.parameters(), lr=args['lr'])
    mse = nn.MSELoss()
    rect_flow = rectified_flow(input_dim = args['input_dim'], time_steps = time_steps,
                                tiling = args['tiling'], device=device)
    logger = SummaryWriter(os.path.join("runs", args['run_name']))
    l = len(dataloader)

    loss_SB_curve = []
    t = rect_flow.sample_timesteps()  # Timesteps on the same device

    # Mini-batch gradient descent
    for epoch in range(args['epochs']):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data, _) in enumerate(pbar):
            print('iteration:', i, 'epoch:', epoch)
            data = data.to(torch.float32)
            data = data.to(device)  # Ensure images are on the device
            
            
            x_t, _ = rect_flow.noise_images(data, t)  # Ensure these tensors are on the correct device
            t_expanded = t.view(len(t), 1, 1).expand(len(t), data.shape[0], 1)
            

            # Blurred score formulation
            predict_SB = model_SB(x_t[:-1, :, :], t_expanded[:-1]) # to avoid score estimation t = 1
            SB = score_blurred(x_t, data, t, sigma_blurring) # does not give the score at t=1
            print('SB.shape, predict_SB.shape', SB.shape, predict_SB.shape)
            loss_SB = mse(SB, predict_SB) 

            optimizer_SB.zero_grad()
            loss_SB.backward()
            optimizer_SB.step()


            
            

            
            loss_SB_curve.append(loss_SB.item())

            pbar.set_postfix( MSE_SB = loss_SB.item())
            logger.add_scalar("MSE_SB", loss_SB.item(), global_step=epoch * l + i)
        # Sample and return images
   
    sampled_data_SB, init_samples_SB = rect_flow.sample_SB(model = model_SB, time_seq = t, target_mean = torch.mean(dataloader.dataset),n=sample_size)
    return  sampled_data_SB, init_samples_SB, loss_SB_curve, rect_flow.traj_SB
         



def emd2_multivariate(data1, data2):
    """
    Computes Earth Mover's Distance (EMD) between two multivariate distributions.
    
    Parameters:
    - data1: np.ndarray of shape (n_samples1, n_features) - Samples from first distribution
    - data2: np.ndarray of shape (n_samples2, n_features) - Samples from second distribution
    
    Returns:
    - emd_value: The computed EMD value between data1 and data2
    """
    # Uniform weights for each sample
    weights1 = np.ones((len(data1),)) / len(data1)
    weights2 = np.ones((len(data2),)) / len(data2)

    # Compute pairwise cost (distance) matrix
    cost_matrix = ot.dist(data1, data2, metric='euclidean')

    # Compute EMD using the optimal transport function from POT
    emd_value = ot.emd2(weights1, weights2, cost_matrix)
    return emd_value




def kl_kde(P, Q):
    kde_p = gaussian_kde(P.T)
    kde_q = gaussian_kde(Q.T)
    return np.mean(np.log(kde_p(P.T)) - np.log(kde_q(P.T)))