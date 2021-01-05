# Deep Helmholtz Decomposition
# Andrew Sosanya, Sam Greydanus | 2020

import numpy as np

def get_dampedspring_data(args):
  x1, x2 = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25)) # coordinates where arrow starts
  dx1_rot, dx2_rot = -x2, x1  # choosing our gradients for our component fields
  dx1_irr, dx2_irr = x1, x2
  dx1 = dx1_rot + dx2_irr  # and adding them together to get our composite
  dx2 = dx1_rot + dx2_irr
  
  # Reshaping our data to fit convention of x = input, y = target
  x = np.concatenate([x1.reshape(-1,1), x2.reshape(-1,1)], axis=1) #axis in both pytorch and numpy
  dx_rot = np.concatenate([dx1_rot.reshape(-1,1), dx2_rot.reshape(-1,1)], axis=1)
  dx_irr = np.concatenate([dx1_irr.reshape(-1,1), dx2_irr.reshape(-1,1)], axis=1)
  dx = dx_rot + dx_irr
  
  # Shuffle the dataset so there aren't any order effects
  shuffle_ixs = np.random.permutation(x.shape[0])
  [x, dx_rot, dx_irr, dx] = [v[shuffle_ixs] for v in [x, dx_rot, dx_irr, dx]]

  # Split the dataset into it's training and testing components.
  #   axes of tensors are [dataset_size, coordinates] (where coordinates = features)
  split_ix = int(x.shape[0] * args.train_split) # train / test split
  data = {'x': x[:split_ix], 'x_test': x[split_ix:], 
          'dx_rot': dx_rot[:split_ix], 'dx_rot_test': dx_rot[split_ix:],
          'dx_irr': dx_irr[:split_ix], 'dx_irr_test': dx_irr[split_ix:],
          'dx': dx[:split_ix], 'dx_test': dx[split_ix:]}

  return data