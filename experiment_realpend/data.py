# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

import os, sys
from urllib.request import urlretrieve
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import read_lipson, str2array

def get_lipson_data(args, save_path=None):
  '''Downloads and formats the datasets provided in the supplementary materials of
  the 2009 Lipson Science article "Distilling Free-Form Natural Laws from
  Experimental Data."
  Link to supplementary materials: https://bit.ly/2JNhyQ8
  Link to article: https://bit.ly/2I2TqXn
  '''
  if save_path is None:
    save_path = './experiment_realpend/'
  url = 'http://science.sciencemag.org/highwire/filestream/590089/field_highwire_adjunct_files/2/'
  os.makedirs(save_path) if not os.path.exists(save_path) else None
  try:
    urlretrieve(url, save_path + '/invar_datasets.zip')
  except:
    print("Failed to download dataset.")
  try:
    data_str = read_lipson(dataset_name="real_pend_h_1", save_path=save_path)
    print("Succeeded at finding and reading dataset.")
  except:
    print("Failed to find/read dataset.")
  state, names = str2array(data_str)

  # estimate dx using finite differences
  data = {k: state[:,i:i+1] for i, k in enumerate(names)}
  x = state[:,2:4]
  dx = (x[1:] - x[:-1]) / (data['t'][1:] - data['t'][:-1])
  dx[:-1] = (dx[:-1] + dx[1:]) / 2  # midpoint rule
  x, t = x[1:], data['t'][1:]

  split_ix = int(state.shape[0] * args.train_split) # train / test split
  data['x'], data['x_test'] = x[:split_ix], x[split_ix:]
  data['t'], data['t_test'] = 0*x[:split_ix,...,:1], 0*x[split_ix:,...,:1] # H = not time varying
  data['dx'], data['dx_test'] = dx[:split_ix], dx[split_ix:]
  data['time'], data['time_test'] = t[:split_ix], t[split_ix:]
  return data


  ### FOR DYNAMICS IN ANALYSIS SECTION ###
def hamiltonian_fn(coords):
  k = 2.4  # this coefficient must be fit to the data
  q, p = np.split(coords,2)
  H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
  return H

def dynamics_fn(t, coords):
  dcoords = autograd.grad(hamiltonian_fn)(coords)
  dqdt, dpdt = np.split(dcoords,2)
  S = -np.concatenate([dpdt, -dqdt], axis=-1)
  return S