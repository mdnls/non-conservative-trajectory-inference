import torch.nn as nn
import torch
import torchsde 

### three models from classic dynamical systems

class LotkaVolterra(nn.Module):
    # predator-prey
    def __init__(self, alpha, beta, gamma, delta, sigma):
        super(LotkaVolterra, self).__init__()
        self.alpha = nn.Parameter(torch.log(torch.tensor(alpha)))
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.delta = nn.Parameter(torch.log(torch.tensor(delta)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        alpha = self.preprocess(self.alpha) # things needs to be positive
        beta = self.preprocess(self.beta)
        gamma = self.preprocess(self.gamma)
        delta = self.preprocess(self.delta)
        y = torch.relu(y) + 1e-10 # avoid going to 0
        dxdt =  alpha * y[:,0] - beta * y[:,0] * y[:,1]
        dydt = delta * y[:,0] * y[:,1] - gamma * y[:,1]
        return torch.stack([dxdt, dydt], dim = 1)
    
    def g(self, t, y):
        sigma = self.preprocess(self.sigma)
        return sigma * torch.relu(y)
    
class LambOseen(nn.Module):
    # simple 2D Lamb-Oseen flow
    def __init__(self, x0,y0, logscale, circulation, logsigma):
        super(LambOseen, self).__init__()
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.logscale = nn.Parameter(torch.tensor(logscale))
        self.circulation = nn.Parameter(torch.tensor(circulation))
        self.logsigma = nn.Parameter(torch.tensor(logsigma))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        scale = torch.exp(self.logscale)
        xx = (y[:,0] - self.x0) * torch.exp(-scale)
        yy = (y[:,1] - self.y0) * torch.exp(-scale)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        dthetadt = 1./r * (1- torch.exp(-r**2))
        dxdt = -dthetadt * torch.sin(theta)
        dydt = dthetadt * torch.cos(theta)
        return self.circulation * torch.stack([dxdt, dydt], dim = 1)
    
    def g(self, t, y):
        return torch.exp(self.logsigma) * torch.ones_like(y)
    

class repressilator(nn.Module):
    # biological clock 
    def __init__(self, beta, n, k, gamma, sigma):
        super(repressilator, self).__init__()
        self.beta = nn.Parameter(torch.log(torch.tensor(beta)))
        self.n = nn.Parameter(torch.log(torch.tensor(n)))
        self.k = nn.Parameter(torch.log(torch.tensor(k)))
        self.gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self.sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.preprocess = torch.exp
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t, y):
        beta = self.preprocess(self.beta)
        n = self.preprocess(self.n)
        k = self.preprocess(self.k)
        gamma = self.preprocess(self.gamma)
        y = torch.relu(y) + 1e-8 # concentration has to be positive
        dxdt = beta/(1.+ (y[:,2]/k) ** n) - gamma * y[:,0]
        dydt = beta/(1.+ (y[:,0]/k) ** n) - gamma * y[:,1]
        dzdt = beta/(1.+ (y[:,1]/k) ** n) - gamma * y[:,2]
        return torch.stack([dxdt, dydt, dzdt], dim = 1)
    def g(self, t,y):
        sigma = self.preprocess(self.sigma)
        return sigma * torch.relu(y)


