import torch
from torch import nn

# https://discuss.pytorch.org/t/vanishing-gradients/46824/7
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

def train_loopGeneralSym(numeric_step, dataloader, model, loss_fn, optimizer, scheduler, sch, params = None):
    for X, y, tau in dataloader:
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

def test_loopGeneralSym(numeric_step, dataloader, model, loss_fn, params = None):
    with torch.no_grad():
        for X, y , tau in dataloader:
            inverse, _ = model.back(X, torch.pow(tau, 2)) # Pass trough inverse model
        
            # Numerical method step
            XX = numeric_step(inverse, tau, params)
            

            pred, _ = model(XX, torch.pow(tau, 2)) # Pass trough original model
            loss = loss_fn(pred, y)

    return loss

def train_loop(dataloader, model, omega, loss_fn, optimizer, scheduler, sch):

    for batch, (X, y, tau) in enumerate(dataloader):
        inverse, _ = model.back(X, tau) # Pass trough inverse model
        
        # Need to do the numerical method now (symplectic Euler)
        XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
            
        
        a = inverse[:, 0, 1] - omega**2*tau.T*inverse[:, 0, 0]
        b = inverse[:, 0, 0] + a[0,  :]*tau.T
        a = a.reshape((1, 1, X.size(dim=0)))
        XX[:, 0, 1] = a
        XX[:, 0, 0] = b
        
        pred, _ = model(XX, tau) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Debug for cheking gradients when there is afear of vanishing/exploding gradients
        #grads = plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

def test_loop(dataloader, model, omega, loss_fn):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            inverse, _ = model.back(X, tau) # Pass trough inverse model
        
            # Need to do the numerical method now (symplectic Euler)
            XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
            
            a = inverse[:, 0, 1] -omega**2*tau.T*inverse[:, 0, 0]
            b = inverse[:, 0, 0] + a[0,  :]*tau.T
            a = a.reshape((1, 1, X.size(dim=0)))
            XX[:, 0, 1] = a
            XX[:, 0, 0] = b

            pred, _ = model(XX, tau) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss

def train_loop_pendulum(dataloader, model, loss_fn, optimizer, scheduler, sch):

    for batch, (X, y, tau) in enumerate(dataloader):
        inverse, _ = model.back(X, tau) # Pass trough inverse model
        
        # Need to do the numerical method now (symplectic Euler)
        XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
        
        a = inverse[:, 0, 1] -tau.T*torch.sin(inverse[:, 0, 0])
        b = inverse[:, 0, 0] + a[0,  :]*tau.T
        a = a.reshape((1, 1, X.size(dim=0)))
        XX[:, 0, 1] = a
        XX[:, 0, 0] = b
        
        pred, _ = model(XX, tau) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Debug for cheking gradients when there is afear of vanishing/exploding gradients
        #grads = plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

def test_loop_pendulum(dataloader, model, loss_fn):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y, tau) in enumerate(dataloader):
            inverse, _ = model.back(X, tau) # Pass trough inverse model
        
            # Need to do the numerical method now (symplectic Euler)
            XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
            
            a = inverse[:, 0, 1] -tau.T*torch.sin(inverse[:, 0, 0])
            b = inverse[:, 0, 0] + a[0,  :]*tau.T
            a = a.reshape((1, 1, X.size(dim=0)))
            XX[:, 0, 1] = a
            XX[:, 0, 0] = b

            pred, _ = model(XX, tau) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss

def train_loop_kepler(dataloader, model, loss_fn, optimizer, scheduler, sch):

    for batch, (X, y, tau) in enumerate(dataloader):
        inverse, _ = model.back(X, tau) # Pass trough inverse model
        
        # Need to do the numerical method now (symplectic Euler)
        XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
        tau_vec = tau.reshape(tau.size(0))
            
        q1 = inverse[:, 0, 0] +tau_vec*inverse[:, 0, 2]
        q2 = inverse[:, 0, 1] +tau_vec*inverse[:, 0, 3]
        d = torch.pow(torch.pow(q1, 2) +torch.pow(q2, 2), 1.5)
        p1 = inverse[:, 0, 2] -tau_vec*q1/d
        p2 = inverse[:, 0, 3] -tau_vec*q2/d

        XX[:, 0, 0] = q1
        XX[:, 0, 1] = q2
        XX[:, 0, 2] = p1
        XX[:, 0, 3] = p2

        pred, _ = model(XX, tau) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Debug for cheking gradients when there is afear of vanishing/exploding gradients
        #grads = plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

def test_loop_kepler(dataloader, model, loss_fn):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            inverse, _ = model.back(X, tau) # Pass trough inverse model
        
            # Need to do the numerical method now (symplectic Euler)
            XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
            tau_vec = tau.reshape(tau.size(0))
                
            q1 = inverse[:, 0, 0] +tau_vec*inverse[:, 0, 2]
            q2 = inverse[:, 0, 1] +tau_vec*inverse[:, 0, 3]
            d = torch.pow(torch.pow(q1, 2)+torch.pow(q2, 2), 1.5)
            p1 = inverse[:, 0, 2] -tau_vec*q1/d
            p2 = inverse[:, 0, 3] -tau_vec*q2/d

            XX[:, 0, 0] = q1
            XX[:, 0, 1] = q2
            XX[:, 0, 2] = p1
            XX[:, 0, 3] = p2

            pred, _ = model(XX, tau) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss

def train_loop_classic(dataloader, model, loss_fn, optimizer, scheduler, sch):

    for batch, (X, y, tau) in enumerate(dataloader): 
        pred, _ = model(X, tau) # Pass trough original model
        loss = loss_fn(pred, y)

        error = loss.detach()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Debug for cheking gradients when there is afear of vanishing/exploding gradients
        #grads = plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

def test_loop_classic(dataloader, model, loss_fn):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            pred, _ = model(X, tau) # Pass trough original model
            loss += loss_fn(pred, y).item()

    loss /= batches

    return loss