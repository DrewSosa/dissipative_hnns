# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

from ..utils import from_pickle
import numpy as np

def to_cos_embedding(t, w=10):
    tsteps = len(t)
    t = t.reshape(-1,1).repeat(w,-1)
    ws = np.linspace(1,20,w-1).reshape(1,-1).repeat(tsteps,0)
#     ws = np.linspace(0.9,12,w-1).reshape(1,-1).repeat(tsteps,0)
    t[:,1:] = np.cos(t[:,1:]*ws)
    return t

def get_ocean_data(args):
    atlantic = from_pickle('./experiment_ocean/oscar_vel2020_preprocessed.pkl')
    atlantic[:,-1] = -1 * atlantic[:,-1]  # flip y axes; we will be plotting with flipped y coordinates

    cropped_atlantic = atlantic[:,:,90:120,90:120]
    cropped_atlantic = np.nan_to_num(cropped_atlantic) #change all NaNs to 0
    x, y = np.meshgrid(range(cropped_atlantic.shape[2]), range(cropped_atlantic.shape[3]))
    xy_max = 1. * max(x.max(), y.max())

    # Reshaping our data to fit convention of x = input, y = target
    tsteps = 69
    
    t = np.linspace(-1,1,tsteps) # [tsteps]
    t = to_cos_embedding(t, w=10) # [tsteps, w]
    t = t.reshape(-1,1,1,10).repeat(cropped_atlantic.shape[-2]*cropped_atlantic.shape[-1],1)
    t = t.reshape(-1,10)
    x = np.concatenate([x.reshape(1,-1,1).repeat(tsteps,0),
                        y.reshape(1,-1,1).repeat(tsteps,0)], axis=-1)
    x = x.reshape(-1,2) # mix time and batch dimension
    x = 3 * x / xy_max  # divide by the maximum coordinate value for xs, ys
    x -= x.mean()
    
    y = np.concatenate([cropped_atlantic[:tsteps,0,:,:].reshape(tsteps,-1,1),
                        cropped_atlantic[:tsteps,1,:,:].reshape(tsteps,-1,1)], axis=-1)
    y = y.reshape(-1,2) # mix time and batch dimension
    y = (y-y.mean()) / (y.std())  # simple normalization

    # Shuffle the dataset so there aren't any order effects
    shuffle_ixs = np.random.permutation(x.shape[0])
    x, y, t = x[shuffle_ixs], y[shuffle_ixs], t[shuffle_ixs]

    # Split the dataset into it's training and testing components.
    #   axes of tensors are [dataset_size, coordinates] (where coordinates = features)
    split_ix = int(x.shape[0]*args.train_split) # train / test split
    data = {'x': x[:split_ix], 'x_test': x[split_ix:], 
            't': t[:split_ix], 't_test': t[split_ix:], 
            'dx': y[:split_ix], 'dx_test': y[split_ix:],
            'atlantic': atlantic}
    return data