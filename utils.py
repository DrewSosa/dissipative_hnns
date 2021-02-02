# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

import numpy as np
import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


def integrate_model(model, t_span, y0, fun=None, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  fun = default_fun if fun is None else fun
  return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def L2_loss(u, v):
  return (u-v).pow(2).mean()


def read_lipson(dataset_name, save_path):
  desired_file = dataset_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_path)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)


def str2array(string):
  lines = string.split('\\n')
  names = lines[0].strip("b'% \\r").split(' ')
  dnames = ['d' + n for n in names]
  names = ['trial', 't'] + names + dnames
  data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]

  return np.asarray(data), names


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing



def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
    '''Given a three dimensional array [frames, height, width], make
    a gif and save it.'''
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    for i in range(len(frames)):
        im = (frames[i].clip(-.5,.5) + .5)*255
        im[divider,:] = 0
        im[divider + 1,:] = 255
        if pixels is not None:
          im = scipy.misc.imresize(im, pixels)
        scipy.misc.imsave(temp_dir + '/f_{:04d}.png'.format(i), im)

    images = []
    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(temp_dir, file_name)
            images.append(imageio.imread(file_path))
    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)

    shutil.rmtree(temp_dir) # remove all the images
    return png_save_path