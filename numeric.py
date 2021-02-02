# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus

from scipy import interpolate
import numpy as np
from functools import partial
RHO = 0.75
GRIDSIZE = 20

# A numerical method (doesn't use NNs) for performing helmholtz decompositions

# Functions for interpolating data so as to obtain an NxM matrix
def get_interp_model(x, dx, method='nearest'):
    return partial(interpolate.griddata, x, dx, method=method)
    
def coords2fields(x, dx, hw=None, replace_nans=True, method='nearest', verbose=True):
    '''The x and dx coords are shuffled along the batch dimension. We need to
    run an interpolation routine in order to obtain a tensor representation of the field.'''
    if hw is None:
        h = w = GRIDSIZE # assume h=w and x=h*w
        if verbose: print('Using gridsize={}'.format(GRIDSIZE))
    else:
        (h, w) = hw
    xx = np.linspace(x[:,0].min(), x[:,1].max(), w)
    yy = np.linspace(x[:,0].min(), x[:,1].max(), h)
    x_field = np.stack(np.meshgrid(xx, yy), axis=-1)
    
    interp_model = get_interp_model(x, dx, method=method)
    dx_field = interp_model(x_field)
    if replace_nans:
        dx_field[np.where(np.isnan(dx_field))] = np.nanmean(dx_field)
    return x_field, dx_field

def project(vx, vy):
  """Project the velocity field to be approximately mass-conserving. Technically
  we are finding an approximate solution to the Poisson equation."""
  print(vx.shape)
  p = np.zeros(vx.shape)
  div = -0.5 * (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)
              + np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0))

  for k in range(1000):  # use gauss-seidel to approximately solve linear system
      p = (div + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)
                + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))/4.0

  vx = vx - 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
  vy = vy - 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
  return vx, vy

def approx_helmholtz_decomp(x, dx, **kwargs): # assumes x, dx have axes (batch, coords)
    x_field, dx_field = coords2fields(x, dx, **kwargs)
    dx0, dx1 = dx_field[...,0], dx_field[...,1]
    dx_rot = np.stack(project(dx0, dx1), axis=-1)
    dx_irr = dx_field - dx_rot # Helmholtz-Hodge decomposition identity
    return x_field, dx_field, dx_rot, dx_irr