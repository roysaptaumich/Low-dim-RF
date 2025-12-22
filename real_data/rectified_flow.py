

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from straightness import compute_norm_squared_full_derivative

class RectifiedFlow():
    def __init__(self, device, img_size=32, c_in=3):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.c_in = c_in

    def noise_images(self, x, t):
        z = torch.randn_like(x)
        tt1 = (1-t)[:, None, None, None]
        tt2 = t[:, None, None, None]
        return tt1 * z + tt2 * x, z
    
    def sample_timesteps(self, n):
        return torch.rand(n)
    
    

    def sample(self, model, n, time_steps=100, time_grid = 'geometric', batch_size=128, seed=0):
        print(f"Sampling {n} new images in batches of {batch_size}")
        model.eval()
        samples = []

        # create geometric grid
        if time_grid == 'geometric':
                geom_timegrid = np.zeros(time_steps)
                h = np.power((time_steps - 1)/2, 2/(time_steps-2)) - 1.0
                # First partition [0, 1/2]
                geom_timegrid[0] = 0
                geom_timegrid[1] = 1.0 / (time_steps - 1)
                for j in range(2, (time_steps - 1) // 2 + 1):
                    geom_timegrid[j] = (1 + h) * geom_timegrid[j - 1]

                # Second partition [1/2, 1]
                geom_timegrid[time_steps - 1] = 1.0
                geom_timegrid[time_steps - 2] = 1.0 - 1.0 / (time_steps - 1)
                for j in range(time_steps - 3, (time_steps - 1) // 2 - 1, -1):
                    geom_timegrid[j] = 1.0 - (1 + h) * (1.0 - geom_timegrid[j + 1])

        # Set the seed if provided for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            for batch_start in tqdm(range(0, n, batch_size)):
                current_batch_size = min(batch_size, n - batch_start)

                # Change the seed for each batch to ensure different random starting points
                if seed is not None:
                    torch.manual_seed(seed + batch_start)  # Use batch_start to vary the seed

                # Sample the Gaussian noise
                x = torch.randn((current_batch_size, self.c_in, self.img_size, self.img_size)).to(self.device)

                # Iterate over the RF steps
                for i in range(time_steps):
                    if time_grid == 'geometric':
                        t = (torch.ones(current_batch_size) * geom_timegrid[i]).to(self.device)
                        dt = geom_timegrid[min(i+1, time_steps - 1)] - geom_timegrid[i]
                        x += model(x, t) * dt 
                    
                    if time_grid == 'linear':
                        t = (torch.ones(current_batch_size) * i/time_steps).to(self.device)
                        x += model(x, t)/time_steps

                    

                x.clamp(-1, 1)
                # Append the generated samples
                samples.append(x.cpu())

        return torch.cat(samples)
