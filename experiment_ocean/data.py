# Deep Helmholtz Decomposition
# Andrew Sosanya, Sam Greydanus | 2020

from ..utils import from_pickle
import numpy as np

def get_ocean_data(args):
    atlantic = from_pickle('./experiment_ocean/oscar_vel2020_preprocessed.pkl')
    atlantic = np.nan_to_num(atlantic) #change all NaNs to 0
    atlantic[:,-1] = -1 * atlantic[:,-1]  # flip y axes; we will be plotting with flipped y coordinates
    # atlantic = atlantic[:,:,100:150,100:150]
    # atlantic = atlantic[:,:,100:115,100:115]
    atlantic = atlantic[:,:,90:120,90:120]
    # atlantic[...,:1,:1] = atlantic[...,-1:,-1:] = 0 # set boundary conditions (optional)
    x, y = np.meshgrid(range(atlantic.shape[2]), range(atlantic.shape[3]))
    xy_max = 1. * max(x.max(), y.max())

    # Reshaping our data to fit convention of x = input, y = target
    x = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1) 
    y = np.concatenate([atlantic[0,0,:,:].reshape(-1,1),
                      atlantic[0,1,:,:].reshape(-1,1)], axis=1)

    # Normalize inputs and outputs (putting them all roughly in [-1,1])
    x = 3 * x/xy_max ; y = 3 * y/xy_max  # divide by the maximum coordinate value
    x -= x.mean() ; y -= y.mean() # center the coodinates about zero
    y = (y-y.mean()) / (y.std())  # simple normalization


    # Shuffle the dataset so there aren't any order effects
    shuffle_ixs = np.random.permutation(x.shape[0])
    x, y = x[shuffle_ixs], y[shuffle_ixs]

    # Split the dataset into it's training and testing components.
    #   axes of tensors are [dataset_size, coordinates] (where coordinates = features)
    split_ix = int(x.shape[0]*args.train_split) # train / test split
    data = {'x': x[:split_ix], 'x_test': x[split_ix:], 
          'dx': y[:split_ix], 'dx_test': y[split_ix:]}
    return data