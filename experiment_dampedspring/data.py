# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

import numpy as np
import autograd

RHO = 2.0
GRIDSIZE = 25

def hamiltonian_fn(coords):
    q, p = coords[...,0,:], coords[...,1,:] # assume axes [...,pq,xyz]
    H = (p**2).sum() + (q**2).sum() # spring hamiltonian (linear oscillator)
    return H

def analytic_model(coords, t=None, rho=RHO, get_separate=False):
    coords = np.array(coords)
    added_xyz_axis = False  # a bit hacky
    if coords.shape[-1] == 2: # in the case that pq axes is last, add another axis
        coords = coords[...,None]
        added_xyz_axis = True
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = dcoords[...,0:1,:], dcoords[...,1:2,:] # assume axes [...,pq,xyz]
    S = np.concatenate([dpdt, -dqdt], axis=-2)      # conservative (irrotational) component
    R = -rho * coords ;  R[...,1:,:] = 0    # dissipative component (derivative of Rayleigh function)
    if added_xyz_axis:
        S, R = S[...,0], R[...,0]
    return np.stack([S, R]) if get_separate else S + R

def get_dampedspring_data(args, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=GRIDSIZE, rho=RHO):
    
    # meshgrid to get a 2D field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    x = np.stack([b.flatten(), a.flatten()]).T  # axes are [batch, coordinate] where coordinate=(x,y)

    # get vector directions corresponding to positions on the field
    dx_rot, dx_irr = analytic_model(x, t=None, rho=rho, get_separate=True)
    dx = dx_irr + dx_rot
    
    # Shuffle indices
    shuffle_ixs = np.random.permutation(x.shape[0])
    [x, dx, dx_irr, dx_rot] = [v[shuffle_ixs] for v in [x, dx, dx_irr, dx_rot]]
  
    # Construct the dataset
    split_ix = int(x.shape[0] * args.train_split) # train / test split
    t = np.zeros_like(x[...,:1])
    data = {'x': x[:split_ix], 'x_test': x[split_ix:],
            't': t[:split_ix], 't_test': t[split_ix:],
            'dx': dx[:split_ix], 'dx_test': dx[split_ix:],
            'dx_irr': dx_irr[:split_ix], 'dx_irr_test': dx_irr[split_ix:],
            'dx_rot': dx_rot[:split_ix], 'dx_rot_test': dx_rot[split_ix:],
            'meta': locals()}
    return data