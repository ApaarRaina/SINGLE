import numpy as np
import torch
import torch.nn as nn

class Diffusion(nn.Module):

    def __init__(self,timesteps=1000):
        super().__init__()
        self.timesteps=timesteps
        self.register_buffer("beta", self.cosine_beta_schedule())
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0)) 
    
    def cosine_beta_schedule(self, s=0.008):
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps)

        alpha_bar = torch.cos(
            ((t / self.timesteps) + s) / (1 + s) * torch.pi / 2
        ) ** 2

        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return beta.clamp(0.0001, 0.9999)
    
    def linear_beta_schedule(self):
        scale = 1000 / self.timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
    
    def sample(self,x0,t,noise):
         sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
         sqrt_1m_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
         return sqrt_alpha_bar * x0 + sqrt_1m_alpha_bar * noise
    
    def forward(self,img):
        b, c, h, w = img.shape
        t = torch.randint(0, self.timesteps,(b,) ,device=img.device).long()
        noise = torch.randn_like(img) 
        noised_image = self.sample(img,t,noise)
        return noised_image , noise , t


