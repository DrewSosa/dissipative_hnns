# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus

import torch
import numpy as np
import os, copy, time

# simplifies accessing the hyperparameters.
class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
    
def get_args(as_dict=False):
  arg_dict = {'input_dim': 3,
              'hidden_dim': 256, # capacity
              'output_dim': 2,
              'learning_rate': 1e-2, 
              'test_every': 100,
              'print_every': 200,
              'batch_size': 128,
              'train_split': 0.80,  # train/test dataset percentage
              'total_steps': 5000,  # because we have a synthetic dataset
              'device': 'cuda', # {"cpu", "cuda"} for using GPUs
              'seed': 42,
              'as_separate': False,
              'decay': 0}
  return arg_dict if as_dict else ObjectView(arg_dict)


def get_batch(v, step, args):  # helper function for moving batches of data to/from GPU
  dataset_size, num_features = v.shape
  bix = (step*args.batch_size) % dataset_size
  v_batch = v[bix:bix + args.batch_size, :]  # select next batch
  return torch.tensor(v_batch, requires_grad=True,  dtype=torch.float32, device=args.device)


def train(model, args, data):
  """ General training function"""
  model = model.to(args.device)  # put the model on the GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)  # setting the Optimizer

  model.train()     # doesn't make a difference for now
  t0 = time.time()  # logging the time
  results = {'train_loss':[], 'test_loss':[], 'test_acc':[], 'global_step':0}  # Logging the results

  for step in range(args.total_steps):  # training loop 

    x, t, dx = [get_batch(data[k], step, args) for k in ['x', 't', 'dx']]
    dx_hat = model(x, t=t)  # feeding forward
    loss = (dx-dx_hat).pow(2).mean()  # L2 loss function
    loss.backward(retain_graph=False); optimizer.step(); optimizer.zero_grad()  # backpropogation

    results['train_loss'].append(loss.item())  # logging the training loss

    # Testing our data with desired frequency (test_every)
    if step % args.test_every == 0:

      x_test, t_test, dx_test = [get_batch(data[k], step=0, args=args)
                          for k in ['x_test', 't_test', 'dx_test']]
      dx_hat_test = model(x_test, t=t_test)  # testing our model. 
      test_loss = (dx_test-dx_hat_test).pow(2).mean().item() # L2 loss

      results['test_loss'].append(test_loss)
    if step % args.print_every == 0:
      print('step {}, dt {:.3f}, train_loss {:.2e}, test_loss {:.2e}'
            .format(step, time.time()-t0, loss.item(), test_loss)) #.item() is just the integer of PyTorch scalar. 
      t0 = time.time()
  model = model.cpu()
  return results