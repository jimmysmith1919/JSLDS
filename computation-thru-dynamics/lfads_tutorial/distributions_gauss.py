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


"""Distributions used in the LFADS tutorial."""


from __future__ import print_function, division, absolute_import

import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from tensorflow_probability.substrates import jax as tfp 

import lfads_tutorial.utils as utils


tfd = tfp.distributions

def poisson_log_likelihood(x, log_rate):
  """Compute the log likelihood under Poisson distribution

    log poisson(k, r) = log(r^k * e^(-r) / k!)
                      = k log(r) - r - log k!
    log poisson(k, r=exp(l)) = k * l - exp(l) - lgamma(k + 1)

  Args:
    x: binned spike count data.
    log_rate: The (log) rate that define the likelihood of the data
      under the LFADS model.

  Returns:
    The log-likelihood of the data under the model (up to a constant factor).
  """
  # TODO(sussillo): implement full log-likelihood when lgamma is available.
  # Will likely have to be careful about float->int casting then.
  return x * log_rate - np.exp(log_rate) #- np.lgamma(x + 1)


def diag_gaussian_sample(key, mean, logvar, varmin=1e-16):
  """x ~ N(mean, exp(logvar))

  Arguments:
    key: random.PRNGKey for random bits
    mean: np array, mean of gaussian
    logvar: np array, log variance of gaussian
    varmin: minimum variance allowed, helps with numerical stability

  Returns:
    np array samples from the gaussian of the same size as mean and logvar"""
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return mean + np.exp(0.5*logvar_wm) * random.normal(key, mean.shape)


def diag_gaussian_log_likelihood(z, mean=0.0, logvar=0.0, varmin=1e-16):
  """Log-likelihood under a Gaussian distribution with diagonal covariance.
     Returns the log-likelihood for each dimension.

  Args:
    z: The value to compute the log-likelihood.
    mean: The mean of the Gaussian
    logvar: The log variance of the Gaussian.
    varmin: Minimum variance allowed (numerically useful).

  Returns:
    The log-likelihood under the Gaussian model.
  """
  logvar_wm = np.log(np.exp(logvar) + varmin)
  return (-0.5 * (logvar_wm + np.log(2*np.pi) +
                  np.square((z-mean)/( np.exp(0.5*(logvar_wm))))))


def laplace_log_likelihood(z, mean=0.0, b=1.0, bmin=1e-16):
  """Log-likelihood under a laplace distribution.
     Returns the log-likelihood for each dimension.

  Args:
    z: The value to compute the log-likelihood.
    mean: The mean of the laplce
    b: The scale of the laplace.
    bmin: Minimum b allowed (numerically useful).

  Returns:
    The log-likelihood under the laplace  model.
  """
  b = b + bmin
  return -np.log(2*b)-np.abs(z-mean)/b

def laplace_sample(key, mean, b, bmin=1e-16):
  """x ~ Lap(mean, b)

  Arguments:
    key: random.PRNGKey for random bits
    mean: np array, mean of gaussian
    b: np array, scales of laplace
    bmin: minimum b allowed, helps with numerical stability

  Returns:
    np array samples from the gaussian of the same size as mean and logvar"""

  lap = tfd.Laplace(loc=mean, scale = b)
  s = tfd.Sample(lap, sample_shape = mean.shape)
  return s.sample(key)
  
  



def kl_gauss_gauss(z_mean, z_logvar, prior_params, varmin=1e-16):
  """Compute the KL divergence between two diagonal Gaussian distributions.
            KL(q||p) = E_q[log q(z) - log p(z))]
   Args:
    z_mean: mean of posterior, z ~ q(z|x)
    z_logvar: logvar of posterior
    prior_z_mean: mean of prior, z ~ p(z)
    prior_z_logvar: logvar of prior
    varmin: minimum variance allowed, useful for numerical stability

    Returns:
      np array of KL, computed analytically, same size as z_mean
  """
  z_logvar_wm = np.log(np.exp(z_logvar) + varmin)
  prior_mean = prior_params['mean']
  prior_logvar_wm = np.log(np.exp(prior_params['logvar']) + varmin)
  return (0.5 * (prior_logvar_wm - z_logvar_wm
                 + np.exp(z_logvar_wm - prior_logvar_wm)
                 + np.square((z_mean - prior_mean) / np.exp(0.5 * prior_logvar_wm))
                 - 1.0))


# This function is used in the loss, which is already batch aware. But the
# prior parameters are the same across batch, thus in_axes below.
batch_kl_gauss_gauss = vmap(kl_gauss_gauss, in_axes=(0, 0, None, None))




#laplace_log_likelihood(z, mean=0.0, b=1.0, bmin=1e-16):
"""
def kl_gauss_laplace(key, z_mean_t, z_logvar_t, lap_params, varmin=1e-16):
  ll = diag_gaussian_log_likelihood
  lap_ll = laplace_log_likelihood
  sample = diag_gaussian_sample
  nkeys = z_mean_t.shape[0]
  key, skeys = utils.keygen(key, nkeys)

  # Convert AR(1) parameters.
  # z_t = c + phi z_{t-1} + eps, eps \in N(0, noise var)
  lap_mean = lap_params['mean']
  lap_b =  lap_params['b']+varmin


  # Sample first AR(1) step according to process variance.
  z0 = sample(next(skeys), z_mean_t[0], z_logvar_t[0], varmin)
  logq = ll(z0, z_mean_t[0], z_logvar_t[0], varmin)
  logp = lap_ll(z0, lap_mean, lap_b, 0.0)
  z_last = z0

  # Sample the remaining time steps with adjusted mean and noise variance.
  for z_mean, z_logvar in zip(z_mean_t[1:], z_logvar_t[1:]):
    z = sample(next(skeys), z_mean, z_logvar, varmin)
    logq += ll(z, z_mean, z_logvar, varmin)
    logp += lap_ll(z, lap_mean, lap_b, 0.0)
    z_last = z

  kl = logq - logp
  return kl
"""
def kl_gauss_laplace(key, z_mean_t, z_logvar_t, lap_params, varmin=1e-16):
  """KL using samples for multi-dim gaussian (thru time) and AR(1) process.
  To sample KL(q||p), we sample
        ln q - ln p
  by drawing samples from q and averaging. q is multidim gaussian, p
  is AR(1) process.

  Arguments:
    key: random.PRNGKey for random bits
    z_mean_t: np.array of means with leading dim being time
    z_logvar_t: np.array of log vars, leading dim is time
    ar1_params: dictionary of ar1 parameters, log noise var and autocorr tau
    varmin: minimal variance, useful for numerical stability

  Returns:
    sampled KL divergence between
  """
  ll = diag_gaussian_log_likelihood
  lap_ll = laplace_log_likelihood
  sample = diag_gaussian_sample
  nkeys = z_mean_t.shape[0]
  key, skeys = utils.keygen(key, nkeys)

  # Convert AR(1) parameters.
  # z_t = c + phi z_{t-1} + eps, eps \in N(0, noise var)
  lap_mean = lap_params['mean']
  lap_b =  lap_params['b']+varmin


  # Sample first AR(1) step according to process variance.
  z0 = sample(next(skeys), z_mean_t[0], z_logvar_t[0], varmin)
  logq = ll(z0, z_mean_t[0], z_logvar_t[0], varmin)
  logp = lap_ll(z0, lap_mean, lap_b, 0.0)
  z_last = z0

  # Sample the remaining time steps with adjusted mean and noise variance.
  for z_mean, z_logvar in zip(z_mean_t[1:], z_logvar_t[1:]):
    z = sample(next(skeys), z_mean, z_logvar, varmin)
    logq += ll(z, z_mean, z_logvar, varmin)
    logp += lap_ll(z, lap_mean, lap_b, 0.0)
    z_last = z

  kl = logq - logp
  return kl


def kl_gauss_laplace_single_sample(key, z_mean_t, z_logvar_t, lap_params, varmin=1e-16):
    ll = diag_gaussian_log_likelihood
    lap_ll = laplace_log_likelihood
    sample = diag_gaussian_sample

    lap_mean = lap_params['lap_mean']
    lap_b = lap_params['lap_b']

    num_time_samples = z_mean_t.shape[0] 
    keys = random.split(key, num_time_samples)
    time_samples = vmap(sample, (0, 0, 0, None))(keys, z_mean_t, z_logvar_t, varmin)
    logqs = vmap(ll, (0,0,0, None))(time_samples, z_mean_t, z_logvar_t, varmin)
    logps = vmap(lap_ll, (0,None, None, None))(time_samples, lap_mean, lap_b, 0.0)
    return np.sum(logqs-logps)

def kl_gauss_laplace(key, z_mean_t, z_logvar_t, lap_params, varmin=1e-16):
    keys = random.split(key, lap_params['num_samples'])
    kl = np.mean(vmap(kl_gauss_laplace_single_sample,
                      (0, None, None, None, None))(keys, z_mean_t, z_logvar_t,
                                                   lap_params, varmin))
    return kl



# This function is used in the loss, which is already batch aware. But the
# prior parameters are the same across batch, thus in_axes below.
batch_kl_gauss_laplace = vmap(kl_gauss_laplace, in_axes=(0, 0, 0, None, None))


def kl_gauss_ar1(key, z_mean_t, z_logvar_t, ar1_params, varmin=1e-16):
  """KL using samples for multi-dim gaussian (thru time) and AR(1) process.
  To sample KL(q||p), we sample
        ln q - ln p
  by drawing samples from q and averaging. q is multidim gaussian, p
  is AR(1) process.

  Arguments:
    key: random.PRNGKey for random bits
    z_mean_t: np.array of means with leading dim being time
    z_logvar_t: np.array of log vars, leading dim is time
    ar1_params: dictionary of ar1 parameters, log noise var and autocorr tau
    varmin: minimal variance, useful for numerical stability

  Returns:
    sampled KL divergence between
  """
  ll = diag_gaussian_log_likelihood
  sample = diag_gaussian_sample
  nkeys = z_mean_t.shape[0]
  key, skeys = utils.keygen(key, nkeys)

  # Convert AR(1) parameters.
  # z_t = c + phi z_{t-1} + eps, eps \in N(0, noise var)
  ar1_mean = ar1_params['mean']
  ar1_lognoisevar = np.log(np.exp(ar1_params['lognvar'] + varmin))
  phi = np.exp(-np.exp(-ar1_params['logatau']))
  # The process variance a function of noise variance, so I added varmin above.
  # This affects log-likelihood funtions below, also.
  logprocessvar = ar1_lognoisevar - (np.log(1-phi) + np.log(1+phi))

  # Sample first AR(1) step according to process variance.
  z0 = sample(next(skeys), z_mean_t[0], z_logvar_t[0], varmin)
  logq = ll(z0, z_mean_t[0], z_logvar_t[0], varmin)
  logp = ll(z0, ar1_mean, logprocessvar, 0.0)
  z_last = z0

  # Sample the remaining time steps with adjusted mean and noise variance.
  for z_mean, z_logvar in zip(z_mean_t[1:], z_logvar_t[1:]):
    z = sample(next(skeys), z_mean, z_logvar, varmin)
    logq += ll(z, z_mean, z_logvar, varmin)
    logp += ll(z, ar1_mean + phi * z_last, ar1_lognoisevar, 0.0)
    z_last = z

  kl = logq - logp
  return kl


# This function is used in the loss, which is already batch aware. But the
# prior parameters are the same across batch, thus in_axes below.
batch_kl_gauss_ar1 = vmap(kl_gauss_ar1, in_axes=(0, 0, 0, None, None))



def diagonal_gaussian_params(key, n, mean=0.0, var=1.0):
  """Generate parameters for a diagonal gaussian distribution.

  Arguments:
    key: random.PRNGKey for random bits
    n: size of gaussian
    mean: mean of diagonal gaussian
    var: variance of diagonal gaussian

  Returns:
   dict of np arrays for the initial parameters of the gaussian
  """
  return {'mean' : mean * np.ones((n,)),
          'logvar' : np.log(var) * np.ones((n,))}


def ar1_params(key, n, mean, autocorrelation_tau, noise_variance):
  """AR1 model x_t = c + phi x_{t-1} + eps, w/ eps \in N(0, noise_var)

  Model an autoregressive model with a mean, autocorrelation tau and noise
  variance. Under the hood, the autocorrelation tau and noise variance are
  transformed into phi and the process variance, as needed.

  Arguments:
    key: random.PRNGKey for random bits
    n: number of ar1 processes to model
    mean: mean of ar1 process
    autocorrelation_tau: autocorrelation time constant of ar1
    noise_variance: noise variance of ar1

  Returns:
    a dictionary of np arrays for the parameters of the ar 1 process
  """
  return {'mean' : mean * np.ones((n,)),
          'logatau' : np.log(autocorrelation_tau * np.ones((n,))),
          'lognvar' : np.log(noise_variance) * np.ones((n,))}


def lap_params(key, n, mean, b):
  """AR1 model x_t = c + phi x_{t-1} + eps, w/ eps \in N(0, noise_var)

  Model an autoregressive model with a mean, autocorrelation tau and noise
  variance. Under the hood, the autocorrelation tau and noise variance are
  transformed into phi and the process variance, as needed.

  Arguments:
    key: random.PRNGKey for random bits
    n: number of ar1 processes to model
    mean: mean of ar1 process
    autocorrelation_tau: autocorrelation time constant of ar1
    noise_variance: noise variance of ar1

  Returns:
    a dictionary of np arrays for the parameters of the ar 1 process
  """
  return {'mean' : mean * np.ones((n,)),
          'b' : b*np.ones((n,))}
