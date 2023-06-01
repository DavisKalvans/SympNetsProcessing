"""
Neural network class for PyTorch
"""
import torch
from torch import nn
import numpy as np
    
# Symplctic gradient (sigmoid) Module 
class SympGradModule(nn.Module):
    def __init__(self, d, n, L, sigma):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.D = 2*d
        self.L = L
        self.Wp = torch.nn.Parameter(sigma*torch.randn((n, d), dtype = torch.float64))
        self.wp = torch.nn.Parameter(sigma*torch.randn((n, 1), dtype = torch.float64))
        self.bp = torch.nn.Parameter(sigma*torch.zeros((1, n), dtype = torch.float64))
        self.Wq = torch.nn.Parameter(sigma*torch.randn((n, d), dtype = torch.float64))
        self.wq = torch.nn.Parameter(sigma*torch.randn((n, 1), dtype = torch.float64))
        self.bq = torch.nn.Parameter(sigma*torch.zeros((1, n), dtype = torch.float64))

    def forward(self, x, tau):
        """
        Forward function
        """
        sigma = nn.Sigmoid()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        Q = q + torch.matmul(h, torch.matmul(sigma(torch.matmul(
                p, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        P = p - torch.matmul(h, torch.matmul(sigma(torch.matmul(
                Q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        
        
        return torch.cat((Q, P), 2), tau

    def back(self, x, tau):
        """
        Backward function
        """
        sigma = nn.Sigmoid()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        P = p + torch.matmul(h, torch.matmul(sigma(torch.matmul(
                q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        Q = q - torch.matmul(h, torch.matmul(sigma(torch.matmul(
                P, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        
        
        
        return torch.cat((Q, P), 2), tau


# Symplctic gradient (Tanh) Module 
class SympGradModuleTanh(nn.Module):
    def __init__(self, d, n, L):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.D = 2*d
        self.L = L
        sigma = np.sqrt(0.01)
        self.Wp = torch.nn.Parameter(sigma*torch.randn((n, d)))
        self.wp = torch.nn.Parameter(sigma*torch.randn((n, 1)))
        self.bp = torch.nn.Parameter(sigma*torch.zeros((1, n)))
        self.Wq = torch.nn.Parameter(sigma*torch.randn((n, d)))
        self.wq = torch.nn.Parameter(sigma*torch.randn((n, 1)))
        self.bq = torch.nn.Parameter(sigma*torch.zeros((1, n)))

    def forward(self, x, tau):
        """
        Forward function
        """
        sigma = nn.Tanh()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        Q = q + torch.matmul(h, torch.matmul(sigma(torch.matmul(
                p, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        P = p - torch.matmul(h, torch.matmul(sigma(torch.matmul(
                Q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        
        
        return torch.cat((Q, P), 2), tau

    def back(self, x, tau):
        """
        Backward function
        """
        sigma = nn.Tanh()
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        P = p + torch.matmul(h, torch.matmul(sigma(torch.matmul(
                q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        Q = q - torch.matmul(h, torch.matmul(sigma(torch.matmul(
                P, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        
        
        
        return torch.cat((Q, P), 2), tau

class LinSympGradModule(nn.Module):
    def __init__(self, d, n, L):
        super().__init__()
        """
        Weights: W, w and bias vector b
        """
        self.d = d
        self.D = 2*d
        self.L = L
        sigma = np.sqrt(0.01)
        self.Wp = torch.nn.Parameter(sigma*torch.randn((n, d)))
        self.wp = torch.nn.Parameter(sigma*torch.randn((n, 1)))
        self.bp = torch.nn.Parameter(sigma*torch.zeros((1, n)))
        self.Wq = torch.nn.Parameter(sigma*torch.randn((n, d)))
        self.wq = torch.nn.Parameter(sigma*torch.randn((n, 1)))
        self.bq = torch.nn.Parameter(sigma*torch.zeros((1, n)))

    def forward(self, x, tau):
        """
        Forward function
        """
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        Q = q + torch.matmul(h, torch.matmul((torch.matmul(
                p, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        P = p - torch.matmul(h, torch.matmul((torch.matmul(
                Q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        
        
        return torch.cat((Q, P), 2), tau

    def back(self, x, tau):
        """
        Backward function
        """
        
        # q and p share the memory with x !
        q = x[:, :, 0:self.d]
        p = x[:, :, self.d:self.D]
        h = tau/self.L
 
        # Symplectic Euler step
        P = p + torch.matmul(h, torch.matmul((torch.matmul(
                q, self.Wq.T) + self.bq), (self.wq*self.Wq)))
        Q = q - torch.matmul(h, torch.matmul((torch.matmul(
                P, self.Wp.T) + self.bp), (self.wp*self.Wp)))
        
        
        
        return torch.cat((Q, P), 2), tau
      