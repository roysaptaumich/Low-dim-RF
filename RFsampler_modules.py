from rectified_flow.samplers.base_sampler import Sampler
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyEulerSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t
        
        # Compute the velocity field at the current state and time
        v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
        
        # Update the state using the Euler formula
        self.x_t = x_t + (t_next - t) * v_t


class MynewSDESampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t

        # helper paramaters
        sigma_sq = t**2 + (1-t)**2
        sigma_sq_next = t_next**2 + (1-t_next)**2
        R = t/sigma_sq**0.5
        R_next = t_next/(sigma_sq_next)**0.5
        psi = (R**2/R_next**2) * ((1 - R_next**2)/(1-R**2)) * (1 - R**2/R_next**2)
        eta = (1 - R**2/R_next**2)


        # Compute the velocity field at the current state and time
        v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
        
        # Score function term
        s_t = (t * v_t - x_t) / (1-t)

        # Update the state using the Euler formula
        x_t_next = (R_next * sigma_sq_next**0.5/ R) * (x_t/sigma_sq**0.5 + eta * sigma_sq**0.5 * s_t + (psi**0.5) * torch.randn_like(x_t))
        self.x_t = x_t_next 



class MytrueEulerSampler(Sampler):
    def __init__(self, k, d, CENTER, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.d = d
        self.CENTER = CENTER

    def step(self):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t
        
        # Compute the transition matrix
        sigma_sq_t = t**2 + (1-t)**2
        eta_t = t_next - t
        # --- Compute Block 1 (A_1) ---
    
        # Calculate the scalar coefficient for the I_k block:
        # (1 + (eta_i * (2*t_i - 1)) / sigma_i_sq)
        coeff_1 = 1.0 + (eta_t * (2.0 * t - 1.0)) / sigma_sq_t
        CENTER_coeff1 = (1-t)/sigma_sq_t * eta_t
        # Create the k x k identity matrix I_k
        I_k = torch.eye(self.k, device=x_t.device)
        
        # Scale the identity matrix I_k
        A_1 = coeff_1 * I_k
        CENTER_block1 = CENTER_coeff1 * torch.ones_like(x_t[:, :self.k])  # Center block for the first k dimensions
        # --- Compute Block 2 (A_2) ---
    
        # Calculate the scalar coefficient for the I_{d-k} block:
        # (1 - eta_i / (1 - t_i))
        dim_d_minus_k = self.d - self.k
        coeff_2 = 1.0 - eta_t / (1.0 - t)
        CENTER_coeff2 =  1/(1.0 - t) * eta_t
        # Create the (d-k) x (d-k) identity matrix I_{d-k}
        I_d_k = torch.eye(dim_d_minus_k, device=x_t.device)
        
        # Scale the identity matrix I_{d-k}
        A_2 = coeff_2 * I_d_k
        CENTER_block2 = CENTER_coeff2 * torch.ones_like(x_t[:, self.k:])  # Center block for the last d-k dimensions
            
        A_t = torch.block_diag(A_1, A_2)
        CENTER_block = torch.concat([CENTER_block1, CENTER_block2], dim=1)
        
        # Update the state using A_t appied to each sample
        self.x_t =  x_t @ A_t.T +   CENTER_block * self.CENTER

class Mynew_trueSDESampler(Sampler):
    def __init__(self, k, d, CENTER, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.d = d
        self.CENTER = CENTER

    def step(self):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t

        # helper paramaters
        sigma_sq = t**2 + (1-t)**2
        sigma_sq_next = t_next**2 + (1-t_next)**2
        R = t/sigma_sq**0.5
        R_next = t_next/(sigma_sq_next)**0.5
        
        # if np.abs(R - 1.0) < 1e-10: psi = 1e-7
        # else: 
        psi = (R**2/R_next**2) * ((1 - R_next**2)/(1-R**2)) * (1 - R**2/R_next**2)

        eta = (1 - R**2/R_next**2)


        
        # --- Compute Block 1 (A_1) ---
    
        # Calculate the scalar coefficient for the I_k block:
        
        coeff_1 =   - 1.0 / sigma_sq
        CENTER_coeff1 = t/sigma_sq 
        # Create the k x k identity matrix I_k
        I_k = torch.eye(self.k, device=x_t.device)
        
        # Scale the identity matrix I_k
        A_1 = coeff_1 * I_k
        CENTER_block1 = CENTER_coeff1 * torch.ones_like(x_t[:, :self.k])  # Center block for the first k dimensions
        # --- Compute Block 2 (A_2) ---
    
        # Calculate the scalar coefficient for the I_{d-k} block:
        # (1 - eta_i / (1 - t_i))
        dim_d_minus_k = self.d - self.k
        coeff_2 =  - 1/ (1.0 - t)**2
        CENTER_coeff2 =  t/(1.0 - t)**2 
        # Create the (d-k) x (d-k) identity matrix I_{d-k}
        I_d_k = torch.eye(dim_d_minus_k, device=x_t.device)
        
        # Scale the identity matrix I_{d-k}
        A_2 = coeff_2 * I_d_k
        CENTER_block2 = CENTER_coeff2 * torch.ones_like(x_t[:, self.k:])  # Center block for the last d-k dimensions
            
        A_t = torch.block_diag(A_1, A_2)
        #print(f"A_t type : {A_t.dtype}, x_t type: {x_t.dtype}, psi: {psi}")
        CENTER_block = torch.concat([CENTER_block1, CENTER_block2], dim=1)
        
        # Compute the velocity field at the current state and time
        s_t =  x_t @ A_t.T +   CENTER_block * self.CENTER
        
        # Score function term
        #s_t = (t * v_t - x_t) / (1-t)
        #print(f"eta: {eta}, psi: {psi}", "R:", R, "R_next:", R_next, "sigma_sq:", sigma_sq, "sigma_sq_next:", sigma_sq_next)
        # Update the state using the Euler formula
        self.x_t = (R_next * sigma_sq_next**0.5 / R) * (x_t/sigma_sq**0.5 + eta * sigma_sq**0.5 * s_t + (psi**0.5) * torch.randn_like(x_t))

import numpy as np
def geom_timegrid_generator(NUM_STEPS, delta):
    geom_timegrid = np.zeros(NUM_STEPS + 1)
    h = np.exp(2/(NUM_STEPS-2) * np.log(1/(2 * delta))) - 1.0
    # First partition [0, 1/2]
    geom_timegrid[0] = 0
    geom_timegrid[1] = delta
    for j in range(2, NUM_STEPS // 2 + 1):
        geom_timegrid[j] = (1 + h) * geom_timegrid[j - 1]

    # Second partition [1/2, 1]
    geom_timegrid[NUM_STEPS] = 1.0
    geom_timegrid[NUM_STEPS - 1] = 1.0 - delta
    for j in range(NUM_STEPS - 2, NUM_STEPS // 2 - 1, -1):
        geom_timegrid[j] = 1.0 - (1 + h) * (1.0 - geom_timegrid[j + 1])
    geom_timegrid = torch.tensor(geom_timegrid, device = device)

    return geom_timegrid


lin_timegrid_generator = lambda NUM_STEPS: torch.linspace(0,1 , NUM_STEPS+1).to(device)


def MynewSDE_timegrid_generator(NUM_STEPS, c0 = 2, c1 = 2):
    N = NUM_STEPS
    f = c1 * np.log(N)/N
    beta1 = 1/N**c0
    betas = torch.zeros(NUM_STEPS , device=device)
    for i in range(NUM_STEPS):
        if i == 0: betas[i] = beta1
        else: betas[i] = f * min(1, beta1 * (1+f)**i)

    omegas = torch.cumprod(1 - betas, dim=0)
    timegrid_flipped = torch.zeros(NUM_STEPS, device=device)
    for i, omega in enumerate(omegas):
        timegrid_flipped[i] = torch.sqrt(omega)/(torch.sqrt(omega) + torch.sqrt(1 - omega))
    
    time_grid = timegrid_flipped.flip(0)

    return time_grid

  