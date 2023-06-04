import torch
from torch import nn

# https://discuss.pytorch.org/t/vanishing-gradients/46824/7
# Can be used to check for gradient vanishing/exploding
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    return ave_grads, max_grads, layers

# Training epoch with nonsymetrical models (k=1)
def train_loopGeneral(numeric_step, dataloader, model, loss_fn, optimizer, scheduler, sch, params = None):
    for batch, (X, y, tau) in enumerate(dataloader):
        inverse, _ = model.back(X, tau) # Pass trough inverse model

        XX = numeric_step(inverse, tau, params)
        
        pred, _ = model(XX, tau) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

# Validation with nonsymetrical models (k=1)
def test_loopGeneral(numeric_step, dataloader, model, loss_fn, params = None):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            inverse, _ = model.back(X, tau) # Pass trough inverse model
        
            # Numerical method step
            XX = numeric_step(inverse, tau, params)
            

            pred, _ = model(XX, tau) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss

# Training epoch with symetrical models (k=2)
def train_loopGeneralSym(numeric_step, dataloader, model, loss_fn, optimizer, scheduler, sch, params = None):
    for batch, (X, y, tau) in enumerate(dataloader):
        inverse, _ = model.back(X, torch.pow(tau, 2)) # Pass trough inverse model

        XX = numeric_step(inverse, tau, params)
        
        pred, _ = model(XX, torch.pow(tau, 2)) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

# Validation with symetrical models (k=2)
def test_loopGeneralSym(numeric_step, dataloader, model, loss_fn, params = None):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            inverse, _ = model.back(X, torch.pow(tau, 2)) # Pass trough inverse model
        
            # Numerical method step
            XX = numeric_step(inverse, tau, params)
            

            pred, _ = model(XX, torch.pow(tau, 2)) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss