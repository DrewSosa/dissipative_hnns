# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus

import numpy as np

def get_spiral_data(args):
  x,y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .25)) # coordinates where arrow starts
  u_rot, v_rot = -y, x  # choosing our gradients for our component fields
  u_irr, v_irr = x, y
  u = u_rot + u_irr  # and adding them together to get our composite
  v = v_rot + v_irr
  
  # Reshaping our data to fit convention of x = input, y = target
  x = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)], axis=1) #axis in both pytorch and numpy
  y_rot = np.concatenate([u_rot.reshape(-1,1),v_rot.reshape(-1,1)], axis=1)
  y_irr = np.concatenate([u_irr.reshape(-1,1),v_irr.reshape(-1,1)], axis=1)
  y = y_rot + y_irr
  
  # Shuffle the dataset so there aren't any order effects
  shuffle_ixs = np.random.permutation(x.shape[0]) 
  x = x[shuffle_ixs]
  y_rot = y_rot[shuffle_ixs]
  y_irr = y_irr[shuffle_ixs]
  y = y[shuffle_ixs]

  # Split the dataset into it's training and testing components.
  #   axes of tensors are [dataset_size, coordinates] (where coordinates = features)
  split_ix = int(x.shape[0]*args.train_split) # train / test split
  data = {'x': x[:split_ix], 'x_test': x[split_ix:], 
          'y_rot': y_rot[:split_ix], 'y_rot_test': y_rot[split_ix:],
          'y_irr': y_irr[:split_ix], 'y_irr_test': y_irr[split_ix:],
          'y': y[:split_ix], 'y_test': y[split_ix:]}

  return data