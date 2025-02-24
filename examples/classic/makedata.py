import torchsde
import torch
import numpy as np
from models import LambOseen, LotkaVolterra, repressilator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# make some example data with only population level snapshots
## some other data we might consider adding
### 1) some observed trajectories
### 2) some observations on velocity
### 3) something predictive to velocity
   
def make_data(taskname):
    if "LV" in taskname:
        gt = LotkaVolterra(1.0, 0.4, 0.4, 0.1, 0.02)
        N_steps = 10 + 1
        dts = torch.tensor( np.array([i for i in range(N_steps)])).to(device)
        num_samples = 200
        X_0 = torch.rand(num_samples, 2)
        X_0[:,0] = X_0[:,0] * .1 + 5  # prey 
        X_0[:,1] = X_0[:,1] * .1 + 4 # predator
        Xs = [None for _ in range(N_steps)]
        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 2)
            X_0[:,0] = X_0[:,0] * .1 + 5  # prey 
            X_0[:,1] = X_0[:,1] * .1 + 4 # predator
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]


        y0 = torch.rand(1000, 2) # initial point to be used in calculating MMD
        y0[:,0] = y0[:,0] * .1 + 5  # prey 
        y0[:,1] = y0[:,1] * .1 + 4 # predator
        y0 = y0.to(device) 
        
    elif "LambOseen" in taskname:
        X0s = [-3., 1.] # intial point
    
        lamboseen = LambOseen(0., 0., -2., 1., -2.)
        lamboseen.to(device)
        N_steps = 10 + 1
        dts = torch.tensor( np.array([i for i in range(N_steps)])).to(device)

        num_samples = 200
        X_0 = torch.rand(num_samples, 2)
        X_0[:,0] = X_0[:,0] * .1 + X0s[0] # confine the initial points   
        X_0[:,1] = X_0[:,1] * .1 + X0s[1] 
        Xs = [None for _ in range(N_steps)]

        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 2)
            X_0[:,0] = X_0[:,0] * .1 + X0s[0] # confine the initial points   
            X_0[:,1] = X_0[:,1] * .1 + X0s[1] 
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(lamboseen, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]


        y0 = torch.rand(1000, 2) # initial point to be used in calculating MMD
        y0[:,0] = y0[:,0] * .1 + X0s[0]   
        y0[:,1] = y0[:,1] * .1 + X0s[1] 
        y0 = y0.to(device)

    elif "Repressilator" in taskname:
        repressilator_gt = repressilator(10.,3.,1.,1., 0.02)
        repressilator_gt.to(device)
        N_steps = 10 + 1
        dts = torch.tensor( np.array([i for i in range(N_steps)])).to(device)

        num_samples = 200
        X_0 = torch.rand(num_samples, 3)
        X_0[:,0] = X_0[:,0] * .1 + 1   
        X_0[:,1] = X_0[:,1] * .1 + 1 
        X_0[:,2] = X_0[:,2] * .1 + 2 
        Xs = [None for _ in range(N_steps)]

        Xs[0] = X_0.to(device)

        for i in range(N_steps-1):
            X_0 = torch.rand(num_samples, 3)
            X_0[:,0] = X_0[:,0] * .1 + 1   
            X_0[:,1] = X_0[:,1] * .1 + 1 
            X_0[:,2] = X_0[:,2] * .1 + 2 
            X_0 = X_0.to(device)
            with torch.no_grad():
                ys = torchsde.sdeint(repressilator_gt, X_0.to(device), torch.tensor([0, dts[i+1]]).to(device), 
                           method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).
            Xs[i+1] = ys[-1]


        y0 = torch.rand(1000, 3) # initial point to be used in calculating MMD
        y0[:,0] = y0[:,0] * .1 + 1   
        y0[:,1] = y0[:,1] * .1 + 1 
        y0[:,2] = y0[:,2] * .1 + 2 
        y0 = y0.to(device) 
        
    filename = f"./data/{taskname}_data.npz"
    
    np.savez(filename, 
             N_steps = N_steps,
             Xs = torch.stack(Xs).cpu().detach().numpy(),
             y0 = y0.cpu().detach().numpy())
    
def main():
    tasks = ["LV", "LambOseen","Repressilator"]
    for task in tasks:
        make_data(task)

if __name__ == '__main__':
    main()