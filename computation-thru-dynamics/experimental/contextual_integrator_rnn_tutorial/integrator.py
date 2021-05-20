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


"""Routines for creating white noise and integrated white noise."""

from __future__ import print_function, division, absolute_import
from functools import partial
import jax.numpy as np
from jax import jit, vmap
from jax import random
import matplotlib.pyplot as plt
import contextual_integrator_rnn_tutorial.utils as utils


def build_input_and_target_pure_integration(input_params, key):
  """Build white noise input and integration targets."""
  bias_val, stddev_val, T, ntime = input_params
  dt = T/ntime

  # Create the white noise input.
  key, skeys = utils.keygen(key, 3)
  random_sample_1x2 = random.normal(next(skeys), shape=(1, 2))
  bias_1x2 = bias_val * 2.0 * (random_sample_1x2 - 0.5)
  stddev = stddev_val / np.sqrt(dt)
  random_samples_tx2 = random.normal(next(skeys), shape=(ntime, 2))
  noise_tx2 = stddev * random_samples_tx2
  white_noise_tx2 = bias_1x2 + noise_tx2

  # The context signal a hot one for the duration of the trial.
  con1_tx2 = np.concatenate((np.ones((ntime,1)), np.zeros((ntime,1))), axis=1)
  con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.ones((ntime,1))), axis=1)
  context = random.bernoulli(next(skeys))
  context_tx2 = np.where(context, con1_tx2, con2_tx2)
  
  # * dt, intentionally left off to get output scaling in O(1).
  targets_t = np.where(context,
                       np.cumsum(white_noise_tx2[:,0]),
                       np.cumsum(white_noise_tx2[:,1]))
  inputs_tx4 = np.concatenate((white_noise_tx2, context_tx2), axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  return inputs_tx4, targets_tx1

# Now batch it and jit.
build_input_and_target = build_input_and_target_pure_integration

#build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
#build_inputs_and_targets_jit = jit(build_inputs_and_targets,
#                                   static_argnums=(0,))

def build_inputs_and_targets(input_params, keys):
  f = partial(build_input_and_target, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)

build_inputs_and_targets_jit = jit(build_inputs_and_targets, static_argnums=(0,))

def build_input_and_target_pure_integration_fix_bias(input_params, key):
  """Build white noise input and integration targets."""
  bias_val, stddev_val, T, ntime = input_params
  dt = T/ntime

  # Create the white noise input.


  
  key, skeys = utils.keygen(key, 3)
  #random_sample_1x2 = random.normal(next(skeys), shape=(1, 2))
  bias_1x2 = bias_val#bias_val * 2.0 * (random_sample_1x2 - 0.5)
  stddev = stddev_val / np.sqrt(dt)
  random_samples_tx2 = random.normal(next(skeys), shape=(ntime, 2))
  noise_tx2 = stddev * random_samples_tx2
  white_noise_tx2 = bias_1x2 + noise_tx2

  # The context signal a hot one for the duration of the trial.
  con1_tx2 = np.concatenate((np.ones((ntime,1)), np.zeros((ntime,1))), axis=1)
  con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.ones((ntime,1))), axis=1)
  context = random.bernoulli(next(skeys))
  context_tx2 = np.where(context, con1_tx2, con2_tx2)
  
  # * dt, intentionally left off to get output scaling in O(1).
  targets_t = np.where(context,
                       np.cumsum(white_noise_tx2[:,0]),
                       np.cumsum(white_noise_tx2[:,1]))
  inputs_tx4 = np.concatenate((white_noise_tx2, context_tx2), axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  return inputs_tx4, targets_tx1

# Now batch it and jit.
build_input_and_target_fix_bias = build_input_and_target_pure_integration_fix_bias

#build_inputs_and_targets = vmap(build_input_and_target, in_axes=(None, 0))
#build_inputs_and_targets_jit = jit(build_inputs_and_targets,
#                                   static_argnums=(0,))

def build_inputs_and_targets_fix_bias(input_params, keys):
  f = partial(build_input_and_target_fix_bias, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)

build_inputs_and_targets_fix_bias_jit = jit(build_inputs_and_targets_fix_bias, static_argnums=(0,))


def build_input_and_target_binary(input_params, key):
  """Build white noise input and integration targets."""
  bias_val, stddev_val, T, ntime = input_params
  dt = T/ntime

  # Create the white noise input.
  key, skeys = utils.keygen(key, 3)
  random_sample_1x2 = random.normal(next(skeys), shape=(1, 2))
  bias_1x2 = bias_val * 2.0 * (random_sample_1x2 - 0.5)
  stddev = stddev_val / np.sqrt(dt)
  random_samples_tx2 = random.normal(next(skeys), shape=(ntime, 2))
  noise_tx2 = stddev * random_samples_tx2
  white_noise_tx2 = bias_1x2 + noise_tx2

  # The context signal a hot one for the duration of the trial.
  con1_tx2 = np.concatenate((np.ones((ntime,1)), np.zeros((ntime,1))), axis=1)
  con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.ones((ntime,1))), axis=1)
  context = random.bernoulli(next(skeys))
  context_tx2 = np.where(context, con1_tx2, con2_tx2)
  
  # * dt, intentionally left off to get output scaling in O(1).
  pure_integration_t = np.where(context,
                       np.cumsum(white_noise_tx2[:,0]),
                       np.cumsum(white_noise_tx2[:,1]))
  #target_value = 2.0 * ((pure_integration_t[-1] > 0.0) - 0.5)
  target_value = 1.0 * (pure_integration_t[-1] > 0.0) 
  targets_tm1 = np.zeros(pure_integration_t.shape[0] - 1)
  #targets_t = np.concatenate((targets_tm1,
  #                             np.array((target_value,), dtype=np.float32)),
  #                            axis=0)

  targets_t = np.array((target_value,), dtype=np.float32)
              
  inputs_tx4 = np.concatenate((white_noise_tx2, context_tx2), axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  return inputs_tx4, targets_tx1




def build_inputs_and_targets_binary(input_params, keys):
  f = partial(build_input_and_target_binary, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)

build_inputs_and_targets_binary_jit = jit(build_inputs_and_targets_binary, static_argnums=(0,))

def build_input_and_target_binary_fix_bias(input_params, key):
  """Build white noise input and integration targets."""
  bias_val, stddev_val, T, ntime = input_params
  dt = T/ntime

  # Create the white noise input.
  key, skeys = utils.keygen(key, 3)
  random_sample_1x2 = random.normal(next(skeys), shape=(1, 2))
  bias_1x2 = bias_val# * 2.0 * (random_sample_1x2 - 0.5)
  stddev = stddev_val / np.sqrt(dt)
  random_samples_tx2 = random.normal(next(skeys), shape=(ntime, 2))
  noise_tx2 = stddev * random_samples_tx2
  white_noise_tx2 = bias_1x2 + noise_tx2

  # The context signal a hot one for the duration of the trial.
  con1_tx2 = np.concatenate((np.ones((ntime,1)), np.zeros((ntime,1))), axis=1)
  con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.ones((ntime,1))), axis=1)
  context = random.bernoulli(next(skeys))
  context_tx2 = np.where(context, con1_tx2, con2_tx2)
  
  # * dt, intentionally left off to get output scaling in O(1).
  pure_integration_t = np.where(context,
                       np.cumsum(white_noise_tx2[:,0]),
                       np.cumsum(white_noise_tx2[:,1]))
  #target_value = 2.0 * ((pure_integration_t[-1] > 0.0) - 0.5)
  target_value = 1.0 * (pure_integration_t[-1] > 0.0) 
  targets_tm1 = np.zeros(pure_integration_t.shape[0] - 1)
  #targets_t = np.concatenate((targets_tm1,
  #                             np.array((target_value,), dtype=np.float32)),
  #                            axis=0)

  targets_t = np.array((target_value,), dtype=np.float32)
              
  inputs_tx4 = np.concatenate((white_noise_tx2, context_tx2), axis=1)
  targets_tx1 = np.expand_dims(targets_t, axis=1)
  return inputs_tx4, targets_tx1




def build_inputs_and_targets_binary_fix_bias(input_params, keys):
  f = partial(build_input_and_target_binary_fix_bias, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)

build_inputs_and_targets_binary_fix_bias_jit = jit(build_inputs_and_targets_binary_fix_bias, static_argnums=(0,))


def plot_batch(ntimesteps, input_bxtxu, target_bxtxo=None, output_bxtxo=None,
               errors_bxtxo=None, ntoplot=1):
  """Plot some white noise / integrated white noise examples."""
  plt.figure(figsize=(10,7))
  plt.subplot(221)
  plt.plot(input_bxtxu[0:ntoplot,:,0].T, 'b')
  plt.plot(input_bxtxu[0:ntoplot,:,2].T, 'k')
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Noise')

  plt.subplot(223)
  plt.plot(input_bxtxu[0:ntoplot,:,1].T, 'b')
  plt.plot(input_bxtxu[0:ntoplot,:,3].T, 'k')  
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Noise')

  plt.subplot(222)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[0:ntoplot,:,0].T);
    plt.xlim([0, ntimesteps-1]);
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("Integration")
  if errors_bxtxo is not None:
    plt.subplot(224)
    plt.plot(errors_bxtxo[0:ntoplot,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("|Errors|")
  plt.xlabel('Timesteps')


