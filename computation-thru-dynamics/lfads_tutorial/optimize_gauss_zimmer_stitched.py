# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Optimization routines for LFADS"""


from __future__ import print_function, division, absolute_import

import datetime
import h5py

import jax.numpy as np
from jax import grad, jit, lax, random, vmap
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
import sklearn

import lfads_tutorial.lfads_gauss_stitched as lfads
import lfads_tutorial.utils as utils

import time


def get_kl_warmup_fun(lfads_opt_hps):
  """Warmup KL cost to avoid a pathological condition early in training.

  Arguments:
    lfads_opt_hps : dictionary of optimization hyperparameters

  Returns:
    a function which yields the warmup value
  """

  kl_warmup_start = lfads_opt_hps['kl_warmup_start']
  kl_warmup_end = lfads_opt_hps['kl_warmup_end']
  kl_min = lfads_opt_hps['kl_min']
  kl_max = lfads_opt_hps['kl_max']
  def kl_warmup(batch_idx):
    progress_frac = ((batch_idx - kl_warmup_start) /
                     (kl_warmup_end - kl_warmup_start))
    kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                         (kl_max - kl_min) * progress_frac + kl_min)
    return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
  return kl_warmup




'''
def get_batch(data, skey, num_steps, batch_size):
  """get batch of data starting at random timesteps """
  keys = random.split(skey, batch_size)
  return vmap(get_trial, (None,0,None))(data, keys, num_steps)

get_batch_jit = jit(get_batch, static_argnums=(2,3))
'''

def get_trial(data, key, num_steps):
    """Get trial starting at random timestep  of num_steps timesteps                            
     data shoule be Txd"""
    idx= random.randint(key, (1,), 0, data.shape[0]-num_steps)[0]
    trial = lax.dynamic_slice(data, [idx,0], [num_steps, data.shape[1]])
    return trial

def get_slice_trial(data, key, num_steps, inds):
  "get data slice from which we will get trial, data should be num_slicesxTxd"
  key, skey = random.split(key)
  idx = random.choice(skey, inds, (1,))[0]
  return get_trial(data[idx], key, num_steps)


def get_batch(data, key, num_steps, inds, batch_size):
  "data should be num_slicesxTxd" 
  keys = random.split(key, batch_size)
  return vmap(get_slice_trial, (None, 0, None, None))(data, keys, num_steps, inds)

get_batch_jit = jit(get_batch, static_argnums=(2,4))



def optimize_lfads_core(key, batch_idx_start, num_batches,
                        update_fun, kl_warmup_fun,
                        opt_state, lfads_hps, lfads_opt_hps, train_data):
  """Make gradient updates to the LFADS model.

  Uses lax.fori_loop instead of a Python loop to reduce JAX overhead. This 
    loop will be jit'd and run on device.

  Arguments:
    init_params: a dict of parameters to be trained
    batch_idx_start: Where are we in the total number of batches
    num_batches: how many batches to run
    update_fun: the function that changes params based on grad of loss
    kl_warmup_fun: function to compute the kl warmup
    opt_state: the jax optimizer state, containing params and opt state
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  key, dkeyg = utils.keygen(key, num_batches) # data
  key, fkeyg = utils.keygen(key, num_batches) # forward pass
  
  # Begin optimziation loop. Explicitly avoiding a python for-loop
  # so that jax will not trace it for the sake of a gradient we will not use.
  def run_update(batch_idx, opt_state):
    kl_warmup = kl_warmup_fun(batch_idx)
    #didxs = random.randint(next(dkeyg), [lfads_hps['batch_size']], 0,
     #                      train_data.shape[0])
    #x_bxt = train_data[didxs].astype(np.float32)

    dkey, dskey = random.split(next(dkeyg))
    worm = random.choice(dskey, np.arange(lfads_hps['num_worms']),
                         p=lfads_hps['worm_probs'] )
    #worm = batch_idx%lfads_hps['num_worms']
    x_bxt = get_batch(train_data[worm], dkey,
                      lfads_hps['ntimesteps'],
                      lfads_hps['train_inds'][worm],
                      lfads_hps['batch_size']).astype(np.float32)
    opt_state = update_fun(batch_idx, opt_state, lfads_hps, lfads_opt_hps,
                           next(fkeyg), x_bxt, kl_warmup, worm)
    return opt_state

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  return lax.fori_loop(lower, upper, run_update, opt_state)


optimize_lfads_core_jit = jit(optimize_lfads_core, static_argnums=(2,3,4,6,7))




def optimize_lfads(key, init_params, lfads_hps, lfads_opt_hps,
                   train_data, eval_data):
  """Optimize the LFADS model and print batch based optimization data.

  This loop is at the cpu nonjax-numpy level.

  Arguments:
    init_params: a dict of parameters to be trained
    lfads_hps: dict of lfads model HPs
    lfads_opt_hps: dict of optimization HPs
    train_data: nexamples x time x ndims np array of data for training

  Returns:
    a dictionary of trained parameters"""
  
  # Begin optimziation loop.
  all_tlosses = {0:[],
                    1:[],
                    2:[],
                    3:[],
                    4:[]}

  all_elosses = {0:[],
                    1:[],
                    2:[],
                    3:[],
                    4:[]}
  train_total = []
  eval_total = []
  
  # Build some functions used in optimization.
  kl_warmup_fun = get_kl_warmup_fun(lfads_opt_hps)
  decay_fun = optimizers.exponential_decay(lfads_opt_hps['step_size'],
                                           lfads_opt_hps['decay_steps'],
                                           lfads_opt_hps['decay_factor'])

  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=lfads_opt_hps['adam_b1'],
                                                     b2=lfads_opt_hps['adam_b2'],
                                                     eps=lfads_opt_hps['adam_eps'])
  opt_state = opt_init(init_params)

  def update_w_gc(i, opt_state, lfads_hps, lfads_opt_hps, key, x_bxt,
                  kl_warmup, worm):
    """Update fun for gradients, includes gradient clipping."""
    params = get_params(opt_state)
    grads = grad(lfads.lfads_training_loss)(params, lfads_hps, key, x_bxt,
                                            kl_warmup,
                                            lfads_opt_hps['keep_rate'], worm)
    clipped_grads = optimizers.clip_grads(grads, lfads_opt_hps['max_grad_norm'])
    return opt_update(i, clipped_grads, opt_state)

 
  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = lfads_hps['batch_size']
  num_batches = lfads_opt_hps['num_batches']
  print_every = lfads_opt_hps['print_every']
  num_opt_loops = int(num_batches / print_every)
  params = get_params(opt_state)
  for oidx in range(num_opt_loops):
    batch_idx_start = oidx * print_every
    start_time = time.time()
    key, tkey, dtkey, dekey = random.split(random.fold_in(key, oidx), 4)
    opt_state = optimize_lfads_core_jit(tkey, batch_idx_start,
                                        print_every, update_w_gc, kl_warmup_fun,
                                        opt_state, lfads_hps, lfads_opt_hps,
                                        train_data)
    batch_time = time.time() - start_time

    # Losses
    params = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    kl_warmup = kl_warmup_fun(batch_idx_start)
    # Training loss
    #didxs = onp.random.randint(0, train_data.shape[0], batch_size)
    #x_bxt = train_data[didxs].astype(onp.float32)
    #key, skey = random.split(key)
    #worm = random.randint(skey, (1,), 0, lfads_hps['num_worms'])[0]
    #worm= onp.random.randint(0,5)
    ttotal = 0
    etotal = 0
    
    for worm in range(lfads_hps['num_worms']):
      key, skey = random.split(key)
      x_bxt = get_batch(train_data[worm], skey,
                        lfads_hps['ntimesteps'],
                        lfads_hps['train_inds'][worm],
                        lfads_hps['batch_size']).astype(np.float32)
    
      key, skey = random.split(key)
      tlosses  =  lfads.lfads_losses_jit(params, lfads_hps, skey, x_bxt,
                                     kl_warmup, 1.0, worm)
      ttotal += tlosses['total']
      
      # Evaluation loss
      key, skey = random.split(key)
      ex_bxt = get_batch_jit(eval_data[worm], skey,
                      lfads_hps['ntimesteps'],
                      lfads_hps['eval_inds'][worm],
                      lfads_hps['batch_size']).astype(onp.float32)

      key, skey = random.split(key)
      elosses = lfads.lfads_losses_jit(params, lfads_hps, skey, ex_bxt,
                                     kl_warmup, 1.0, worm)

      etotal += elosses['total']

      # Saving, printing.
      all_tlosses[worm].append(tlosses)
      all_elosses[worm].append(elosses)
      s1 = "Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}"
      s2 = "    Training losses {:0.0f} = nl_NLL {:0.0f} + approx_NLL {:0.0f} + fp_loss {:0.0f} +taylor_loss {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
      s3 = "        Eval losses {:0.0f} = nl_NLL {:0.0f} + approx_NLL {:0.0f} + fp_loss {:0.0f} +taylor_loss {:0.0f} + KL IC {:0.0f},{:0.0f} + KL II {:0.0f},{:0.0f} + L2 {:0.2f}"
      print('worm: ', worm)
      print(s1.format(batch_idx_start+1, batch_pidx, batch_time,
                   decay_fun(batch_pidx)))
      print(s2.format(tlosses['total'], tlosses['nlog_p_xgz'],
                    tlosses['nlog_p_approx_xgz'], tlosses['fp_loss'],
                    tlosses['taylor_loss'],
                    tlosses['kl_g0_prescale'], tlosses['kl_g0'],
                    tlosses['kl_ii_prescale'], tlosses['kl_ii'],
                    tlosses['l2']))
      print(s3.format(elosses['total'], elosses['nlog_p_xgz'],
                    elosses['nlog_p_approx_xgz'], elosses['fp_loss'],
                    elosses['taylor_loss'],
                    elosses['kl_g0_prescale'], elosses['kl_g0'],
                    elosses['kl_ii_prescale'], elosses['kl_ii'],
                    elosses['l2']))

    train_total.append(ttotal)
    eval_total.append(etotal)
    print('train_total: ', ttotal)
    print('eval_total: ', etotal)
    
  tlosses_thru_training = {}
  elosses_thru_training = {}
  for worm in range(lfads_hps['num_worms']):
    tlosses_thru_training[worm] = utils.merge_losses_dicts(all_tlosses[worm])
    elosses_thru_training[worm] = utils.merge_losses_dicts(all_elosses[worm])

    
  
  optimizer_details = {'tlosses' : tlosses_thru_training,
                       'elosses' : elosses_thru_training,
                       'ttotals' : train_total,
                       'etotals' : eval_total}
    
  return params, optimizer_details


  
