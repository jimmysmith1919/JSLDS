# CopyriAght 2019 Google LLC
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


"""LFADS with JSLDS architecture and loss functions."""


from __future__ import print_function, division, absolute_import
from functools import partial

import jax.numpy as np
from jax import jit, lax, random, vmap, jvp
from jax.experimental import optimizers

import lfads_tutorial.distributions_gauss as dists
import lfads_tutorial.utils as utils

import math



def sigmoid(x):
  return 0.5 * (np.tanh(x / 2.) + 1)


def linear_params(key, o, u, ifactor=1.0):
  """Params for y = w x

  Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor}


def affine_params(key, o, u, ifactor=1.0):
  """Params for y = w x + b

  Arguments:
    key: random.PRNGKey for random bits
    o: output size
    u: input size
    ifactor: scaling factor

  Returns:
    a dictionary of parameters
  """
  key, skeys = utils.keygen(key, 1)
  ifactor = ifactor / np.sqrt(u)
  return {'w' : random.normal(next(skeys), (o, u)) * ifactor,
          'b' : np.zeros((o,))}


def mlp_params(key, nlayers, n):  # pylint: disable=unused-argument
  """Build a very specific multilayer perceptron for picking fixed points.

  Args:
    key: random.PRNGKey for randomness
    nlayers: number of layers in the MLP.
    n: MLP dimension

  Returns:
    List of dictionaries, where list index is the layer (indexed by integer)
      and each dict is the params of the layer, i.e. weights and bias.
  """
  params = [None] * nlayers
  for l in range(nlayers):
    # Below we build against identity, so zeros appropriate here.
    params[l] = {'W': np.zeros([n, n]), 'b': np.zeros(n)}
  return params


def gru_params(key, n, u, ifactor=1.0, hfactor=1.0, hscale=0.0):
  """Generate GRU parameters

  Arguments:
    key: random.PRNGKey for random bits
    n: hidden state size
    u: input size
    ifactor: scaling factor for input weights
    hfactor: scaling factor for hidden -> hidden weights
    hscale: scale on h0 initial condition

  Returns:
    a dictionary of parameters
  """
  key, skeys = utils.keygen(key, 5)
  ifactor = ifactor / np.sqrt(u)
  hfactor = hfactor / np.sqrt(n)

  wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
  wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
  wRUHX = np.concatenate([wRUH, wRUX], axis=1)

  wCH = random.normal(next(skeys), (n,n)) * hfactor
  wCX = random.normal(next(skeys), (n,u)) * ifactor
  wCHX = np.concatenate([wCH, wCX], axis=1)

  return {'h0' : random.normal(next(skeys), (n,)) * hscale,
          'wRUHX' : wRUHX,
          'wCHX' : wCHX,
          'bRU' : np.zeros((n+n,)),
          'bC' : np.zeros((n,))}



def affine(params, x):
  """Implement y = w x + b

  Arguments:
    params: a dictionary of params
    x: np array of input

  Returns:
    np array of output
  """
  return np.dot(params['w'], x) + params['b']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims)
# So map over first dimension to hand t_x_m.
# I.e. if affine yields n_y_1 = dot(n_W_m, m_x_1), then
# batch_affine yields t_y_n.
# And so the vectorization pattern goes for all batch_* functions.
batch_affine = vmap(affine, in_axes=(None, 0))



def mlp_tanh(params, x):
  """Multilayer perceptrain with tanh nonlinearity.

  Args:
    params: dict of params for MLP
    x: input

  Returns:
    hidden state after applying the MLP
  """
  h = x
  for layer in params:
    h = np.tanh(h + np.dot(layer['W'], h) + layer['b'])
  return h


# Sussillo: The relu version just seems to make instability more likely.
def mlp_relu(params, x, b=0.01):
  """Multilayer perceptron with relu nonlinearity.

  Args:
    params: dict of params for MLP
    x: input
    b: static bias for each layer

  Returns:
    hidden state after applying the MLP
  """
  h = x
  for layer in params:
    a = h + np.dot(layer['W'], h) + layer['b'] + b
    h = np.where(a > 0.0, a, 0.0)
  return h


mlp = mlp_tanh
batch_mlp =vmap(mlp, in_axes=(None, 0))




def normed_linear(params, x):
  """Implement y = \hat{w} x, where \hat{w}_ij = w_ij / |w_{i:}|, norm over j

  Arguments:
    params: a dictionary of params
    x: np array of input

  Returns:
    np array of output
  """
  w = params['w']
  w_row_norms = np.sqrt(np.sum(w**2, axis=1, keepdims=True))
  w = w / w_row_norms
  return np.dot(w, x)


# Note not BatchNorm, the neural network regularizer,
# rather just batching the normed linear function above.
batch_normed_linear = vmap(normed_linear, in_axes=(None, 0))


def dropout(x, key, keep_rate):
  """Implement a dropout layer.

  Arguments:
    x: np array to be dropped out
    key: random.PRNGKey for random bits
    keep_rate: dropout rate

  Returns:
    np array of dropped out x
  """
  # The shenanigans with np.where are to avoid having to re-jit if
  # keep rate changes.
  do_keep = random.bernoulli(key, keep_rate, x.shape)
  kept_rates = np.where(do_keep, x / keep_rate, 0.0)
  return np.where(keep_rate < 1.0, kept_rates, x)


# Note that dropout is a feed-forward routine that requires randomness. Thus,
# the keys argument is also vectorized over, and you'll see the correct
# number of keys allocated by the caller.
batch_dropout = vmap(dropout, in_axes=(0, 0, None))


def run_dropout(x_t, key, keep_rate):
  """Run the dropout layer over additional dimensions, e.g. time.

  Arguments:
    x_t: np array to be dropped out
    key: random.PRNGKey for random bits
    keep_rate: dropout rate

  Returns:
    np array of dropped out x
  """
  ntime = x_t.shape[0]
  keys = random.split(key, ntime)
  return batch_dropout(x_t, keys, keep_rate)


def taylor(f, order):
  """Compute nth order Taylor series approximation of f.

  Args:
    f: the function to compute the Taylor series expansion on, with signature
        f:: h, x -> h
    order: the order of the expansion (int)

  Returns:
    order-order Taylor series approximation as a function with signature
      T[f]: h, x -> h
  """

  def jvp_first(f, primals, tangent):
    """Jacobian-vector product of first argument element."""
    x, xs = primals[0], primals[1:]
    return jvp(lambda x: f(x, *xs), (x,), (tangent,))

  def improve_approx(g, k):
    """Improve taylor series approximation step-by-step."""
    return lambda x, v: jvp_first(g, (x, v), v)[1] + f(x) / math.factorial(k)

  approx = lambda x, v: f(x) / math.factorial(order)
  for n in range(order):
    approx = improve_approx(approx, order - n - 1)
  return approx


def taylor_approx_rnn(rnn, params, h_star, x_star, h_approx_tm1, x_t, order):
  xdim = x_t.shape[0]
  hx_star = np.concatenate([h_star, x_star], axis=0)
  hx = np.concatenate([h_approx_tm1, x_t], axis=0)

  Fhx = lambda hx: rnn(params, hx[:-xdim], hx[-xdim:])
  return taylor(Fhx, order)(hx_star, hx - hx_star)


def staylor_rnn(rnn, params, order, h_approx_tm1, x_t):
  """Run the switching taylor rnn."""
  x_star = np.zeros_like(x_t)
  h_star = mlp(params['mlp'], h_approx_tm1)
  F_star = rnn(params['gen'], h_star, x_star)

  # Taylor series expansion includes 0 order, so we subtract it off,
  # using the learned MLP point instead. This makes sense because we
  # expanded around (h*,x*), and if the MLP produces a fixed point (thanks
  # to the fixed point regularization pressure), it is equal to F(h*,x*).
  h_staylor_t = taylor_approx_rnn(rnn, params['gen'], h_star, x_star,
                                  h_approx_tm1, x_t, order)
  h_approx_t = h_staylor_t - F_star + h_star
  #o_approx_t = affine(params['out'], h_approx_t)

  return h_star, F_star, h_approx_t 


def jslds_rnn(rnn, params, h_approx_tm1, x_t):
  return staylor_rnn(rnn, params, 1, h_approx_tm1, x_t)


def gru(params, h, x):
  """Implement the GRU equations.

  Arguments:
    params: dictionary of GRU parameters
    h: np array of  hidden state
    x: np array of input

  Returns:
    np array of hidden state after GRU update"""
  bfg = 0.5
  hx = np.concatenate([h, x], axis=0)
  ru = np.dot(params['wRUHX'], hx) + params['bRU']
  r, u = np.split(ru, 2, axis=0)
  u = u + bfg
  r = sigmoid(r)
  u = sigmoid(u)
  rhx = np.concatenate([r * h, x])
  c = np.tanh(np.dot(params['wCHX'], rhx) + params['bC'])
  return u * h + (1.0 - u) * c


def make_rnn_for_scan(rnn, params):
  """Scan requires f(h, x) -> h, h, in this application.
  Args: 
    rnn : f with sig (params, h, x) -> h
    params: params in f() sig.

  Returns: 
    f adapted for scan
  """
  def rnn_for_scan(h, x):
    h = rnn(params, h, x)
    return h, h
  return rnn_for_scan


def run_rnn(rnn_for_scan, x_t, h0):
  """Run an RNN module forward in time.

  Arguments:
    rnn_for_scan: function for running RNN one step (h, x) -> (h, h)
      The params already embedded in the function.
    x_t: np array data for RNN input with leading dim being time
    h0: initial condition for running rnn

  Returns:
    np array of rnn applied to time data with leading dim being time"""
  _, h_t = lax.scan(rnn_for_scan, h0, x_t)
  return h_t


def run_bidirectional_rnn(params, fwd_rnn, bwd_rnn, x_t):
  """Run an RNN encoder backwards and forwards over some time series data.

  Arguments:
    params: a dictionary of bidrectional RNN encoder parameters
    fwd_rnn: function for running forward rnn encoding
    bwd_rnn: function for running backward rnn encoding
    x_t: np array data for RNN input with leading dim being time

  Returns:
    tuple of np array concatenated forward, backward encoding, and
      np array of concatenation of [forward_enc(T), backward_enc(1)]
  """
  fwd_rnn_scan = make_rnn_for_scan(fwd_rnn, params['fwd_rnn'])
  bwd_rnn_scan = make_rnn_for_scan(bwd_rnn, params['bwd_rnn'])

  fwd_enc_t = run_rnn(fwd_rnn_scan, x_t, params['fwd_rnn']['h0'])
  bwd_enc_t = np.flipud(run_rnn(bwd_rnn_scan, np.flipud(x_t),
                                params['bwd_rnn']['h0']))
  full_enc = np.concatenate([fwd_enc_t, bwd_enc_t], axis=1)
  enc_ends = np.concatenate([bwd_enc_t[0], fwd_enc_t[-1]], axis=0)
  return full_enc, enc_ends


def lfads_params(key, lfads_hps):
  """Instantiate random LFADS parameters.

  Arguments:
    key: random.PRNGKey for random bits
    lfads_hps: a dict of LFADS hyperparameters

  Returns:
    a dictionary of LFADS parameters
  """

  #dim_per_worm = lfads_hps['worm_dim']
  key, skeys = utils.keygen(key, 15+2*10)

  data_dim = lfads_hps['data_dim']
  ntimesteps = lfads_hps['ntimesteps']
  enc_dim = lfads_hps['enc_dim']
  con_dim = lfads_hps['con_dim']
  ii_dim = lfads_hps['ii_dim']
  gen_dim = lfads_hps['gen_dim']
  factors_dim = lfads_hps['factors_dim']
  mlp_nlayers = lfads_hps['mlp_nlayers']
  mlp_n = lfads_hps['mlp_n']
  num_worms = lfads_hps['num_worms']
  
  

  in_params = {'W':random.normal(next(skeys), shape=(num_worms, data_dim, factors_dim)),
               'B':np.zeros((num_worms, factors_dim))}
  
  #in_params = {}
  #for i in range(0,num_worms):
  #  in_params[i] = affine_params(next(skeys), factors_dim, data_dim)

  
  ic_enc_params = {'fwd_rnn' : gru_params(next(skeys), enc_dim, factors_dim),
                   'bwd_rnn' : gru_params(next(skeys), enc_dim, factors_dim)}
  gen_ic_params = affine_params(next(skeys), 2*gen_dim, 2*enc_dim) #m,v <- bi
  ic_prior_params = dists.diagonal_gaussian_params(next(skeys), gen_dim, 0.0,
                                                   lfads_hps['ic_prior_var'])
  con_params = gru_params(next(skeys), con_dim, 2*enc_dim + factors_dim)
  con_out_params = affine_params(next(skeys), 2*ii_dim, con_dim) #m,v
  ii_prior_params = dists.lap_params(next(skeys), ii_dim,
                                     lfads_hps['lap_mean'],
                                     lfads_hps['lap_b']
                                     )
  gen_params = gru_params(next(skeys), gen_dim, ii_dim)
  exp_params =  mlp_params(next(skeys), mlp_nlayers, mlp_n)
  factors_params = linear_params(next(skeys), factors_dim, gen_dim)
  #gauss_params = affine_params(next(skeys), 2*data_dim, factors_dim)

  mean_params = {'W':random.normal(next(skeys), shape=(num_worms, factors_dim, data_dim)),
                 'B':np.zeros((num_worms, data_dim))}
  logvar_params = np.zeros((num_worms, data_dim))
  #mean_params = {}
  #logvar_params = {}
  #for i in range(0,num_worms):
  #  mean_params[i] = affine_params(next(skeys), data_dim, factors_dim)
  #  logvar_params[i] = np.zeros((data_dim,))
  
  return {'data_in': in_params,
          'ic_enc' : ic_enc_params,
          'gen_ic' : gen_ic_params, 'ic_prior' : ic_prior_params,
          'con' : con_params, 'con_out' : con_out_params,
          'ii_prior' : ii_prior_params,
          'gen' : gen_params, 'mlp': exp_params,'factors' : factors_params,
          'gauss_out' : mean_params, 'logvar': logvar_params}


def lfads_encode(params, lfads_hps, key, x_t, keep_rate, worm):
  """Run the LFADS network from input to generator initial condition vars.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_t: np array input for lfads with leading dimension being time
    keep_rate: dropout keep rate

  Returns:
    3-tuple of np arrays: generator initial condition mean, log variance
      and also bidirectional encoding of x_t, with leading dim being time
  """
  key, skeys = utils.keygen(key, 3)



  
  # Encode the input
  data_dim = lfads_hps['data_dim']
  factors_dim = lfads_hps['factors_dim']
  w = lax.dynamic_slice(params['data_in']['W'], [worm,0,0],[1,data_dim, factors_dim] )[0]
  b = lax.dynamic_slice(params['data_in']['B'], [worm,0],[1,factors_dim] )[0]
  
  x_t = x_t @ w + b
  x_t = run_dropout(x_t, next(skeys), keep_rate)
  con_ins_t, gen_pre_ics = run_bidirectional_rnn(params['ic_enc'], gru, gru,
                                                 x_t)
  # Push through to posterior mean and variance for initial conditions.
  xenc_t = dropout(con_ins_t, next(skeys), keep_rate)
  gen_pre_ics = dropout(gen_pre_ics, next(skeys), keep_rate)
  ic_gauss_params = affine(params['gen_ic'], gen_pre_ics)
  ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
  return ic_mean, ic_logvar, xenc_t


def lfads_decode_one_step(params, lfads_hps, key, keep_rate, c, f, g, g_approx, xenc, worm):
  """Run the LFADS network from latent variables to log rates one time step.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    keep_rate: dropout keep rate
    c: controller state at time step t-1
    g: generator state at time step t-1
    f: factors at time step t-1
    xenc: np array bidirectional encoding at time t of input (x_t)

  Returns:
    7-tuple of np arrays all with leading dim being time,
      controller hidden state, generator hidden state, factors, 
      inferred input (ii) sample, ii mean, ii log var, log rates
  """
  keys = random.split(key, 3)
  cin = np.concatenate([xenc, f], axis=0)
  c = gru(params['con'], c, cin)
  cout = affine(params['con_out'], c)
  ii_mean, ii_logvar = np.split(cout, 2, axis=0) # inferred input params
  ii = dists.diag_gaussian_sample(keys[0], ii_mean,
                                  ii_logvar, lfads_hps['var_min'])


  #get out params
  data_dim = lfads_hps['data_dim']
  factors_dim = lfads_hps['factors_dim']
  w = lax.dynamic_slice(params['gauss_out']['W'], [worm,0,0],[1,factors_dim, data_dim] )[0]
  b = lax.dynamic_slice(params['gauss_out']['B'], [worm,0],[1,data_dim] )[0]
  
  

  #run the decoder nl_RNN
  g = gru(params['gen'], g, ii)
  g = dropout(g, keys[1], keep_rate)
  f = normed_linear(params['factors'], g)
  #out_mean = affine(params['gauss_out'][worm], f)
  out_mean = f @ w + b
  #out_mean, out_logvar = np.split(gauss_out, 2, axis=0)
  
  #Run the decoder switching RNN
  g_star, F_star, g_approx  = jslds_rnn(gru, params, g_approx, ii)
  g_approx = dropout(g_approx, keys[1], keep_rate)
  f_approx = normed_linear(params['factors'], g_approx)
  #out_mean_approx = affine(params['gauss_out'][worm], f_approx)
  out_mean_approx = f_approx @ w + b
  #out_mean_approx, out_logvar_approx = np.split(gauss_out_approx, 2, axis=0)

  return c, g, f, ii, ii_mean, ii_logvar, out_mean, g_star,\
    F_star, g_approx, f_approx, out_mean_approx
    

def lfads_decode_one_step_scan(params, lfads_hps, keep_rate, worm, state, key_n_xenc):
  """Run the LFADS network one step, prepare the inputs and outputs for scan.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    keep_rate: dropout keep rate
    state: (controller state at time step t-1, generator state at time step t-1, 
            factors at time step t-1)
    key_n_xenc: (random key, np array bidirectional encoding at time t of input (x_t))

  Returns: 2-tuple of state and state plus returned values
    ((controller state, generator state, factors), 
    (7-tuple of np arrays all with leading dim being time,
      controller hidden state, generator hidden state, factors, 
      inferred input (ii) sample, ii mean, ii log var, 
      log rate))
  """
  key, xenc = key_n_xenc
  c, g, f, g_approx = state
  state_and_returns = lfads_decode_one_step(params, lfads_hps, key, keep_rate,
                                            c, f, g, g_approx,xenc, worm)
  c, g, f, ii, ii_mean, ii_logvar, out_mean, g_star,\
    F_star, g_approx, f_approx, out_mean_approx = state_and_returns
  
  
  state = (c, g, f, g_approx)
  return state, state_and_returns


def lfads_decode(params, lfads_hps, key, ic_mean, ic_logvar, xenc_t, keep_rate, worm):
  """Run the LFADS network from latent variables to log rates.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    ic_mean: np array of generator initial condition mean
    ic_logvar: np array of generator initial condition log variance
    xenc_t: np array bidirectional encoding of input (x_t) with leading dim
      being time
    keep_rate: dropout keep rate

  Returns:
    7-tuple of np arrays all with leading dim being time,
      controller hidden state, inferred input mean, inferred input log var,
      generator hidden state, factors and log rates
  """

  ntime = lfads_hps['ntimesteps']
  key, skeys = utils.keygen(key, 2)

  # Since the factors feed back to the controller,
  #    factors_{t-1} -> controller_t -> sample_t -> generator_t -> factors_t
  # is really one big loop and therefor one RNN.
  c0 = params['con']['h0']
  g0 = dists.diag_gaussian_sample(next(skeys), ic_mean, ic_logvar,
                                  lfads_hps['var_min'])
  f0 = np.zeros((lfads_hps['factors_dim'],))

  # Make all the randomness for all T steps at once, it's more efficient.
  # The random keys get passed into scan along with the input, so the input
  # becomes of a 2-tuple (keys, actual input).
  T = xenc_t.shape[0]
  keys_t = random.split(next(skeys), T)

  g02 = g0
  state0 = (c0, g0, f0, g02)
  decoder = partial(lfads_decode_one_step_scan, *(params, lfads_hps, keep_rate, worm))
  _, state_and_returns_t = lax.scan(decoder, state0, (keys_t, xenc_t))
  return state_and_returns_t


def lfads(params, lfads_hps, key, x_t, keep_rate, worm):
  """Run the LFADS network from input to output.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_t: np array of input with leading dim being time
    keep_rate: dropout keep rate

  Returns:
    A dictionary of np arrays of all LFADS values of interest.
  """

  key, skeys = utils.keygen(key, 2)

  ic_mean, ic_logvar, xenc_t = \
      lfads_encode(params, lfads_hps, next(skeys), x_t, keep_rate, worm)


  
  c_t, gen_t, factor_t, ii_t, ii_mean_t, ii_logvar_t, out_mean_t,  g_star_t,\
  F_star_t, g_approx_t, f_approx_t, out_mean_approx_t,  = \
      lfads_decode(params, lfads_hps, next(skeys), ic_mean, ic_logvar,
                   xenc_t, keep_rate, worm)
  
  # As this is tutorial code, we're passing everything around.
  return {'xenc_t' : xenc_t, 'ic_mean' : ic_mean, 'ic_logvar' : ic_logvar,
          'ii_t' : ii_t, 'c_t' : c_t, 'ii_mean_t' : ii_mean_t,
          'ii_logvar_t' : ii_logvar_t, 'gen_t' : gen_t,
          'factor_t' : factor_t,
          'out_mean_t' : out_mean_t, 
          'g_star_t': g_star_t, 'F_star_t': F_star_t,
          'g_approx_t': g_approx_t, 'f_approx_t': f_approx_t,
          'out_mean_approx_t': out_mean_approx_t}


lfads_encode_jit = jit(lfads_encode)
lfads_decode_jit = jit(lfads_decode, static_argnums=(1,))
lfads_jit = jit(lfads, static_argnums=(1,))

# Batching accomplished by vectorized mapping.
# We simultaneously map over random keys for forward-pass randomness
# and inputs for batching.
batch_lfads = vmap(lfads, in_axes=(None, None, 0, 0, None, None))


def lfads_losses(params, lfads_hps, key, x_bxt, kl_scale, keep_rate, worm):
  """Compute the training loss of the LFADS autoencoder

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    keep_rate: dropout keep rate
    kl_scale: scale on KL

  Returns:
    a dictionary of all losses, including the key 'total' used for optimization
  """

  B = lfads_hps['batch_size']
  key, skeys = utils.keygen(key, 2)
  keys_b = random.split(next(skeys), B)
  lfads = batch_lfads(params, lfads_hps, keys_b, x_bxt, keep_rate, worm)

  # Sum over time and state dims, average over batch.
  # KL - g0
  ic_post_mean_b = lfads['ic_mean']
  ic_post_logvar_b = lfads['ic_logvar']
  kl_loss_g0_b = dists.batch_kl_gauss_gauss(ic_post_mean_b, ic_post_logvar_b,
                                            params['ic_prior'],
                                            lfads_hps['var_min'])
  kl_loss_g0_prescale = np.sum(kl_loss_g0_b) / B  
  kl_loss_g0 = kl_scale * kl_loss_g0_prescale
  
  # KL - Inferred input
  ii_post_mean_bxt = lfads['ii_mean_t']
  ii_post_var_bxt = lfads['ii_logvar_t']
  keys_b = random.split(next(skeys), B)
  kl_loss_ii_b = dists.batch_kl_gauss_laplace(keys_b, ii_post_mean_bxt,
                                          ii_post_var_bxt,
                                          params['ii_prior'],
                                          lfads_hps['var_min'])
  kl_loss_ii_prescale = np.sum(kl_loss_ii_b) / B
  kl_loss_ii = lfads_hps['ii_kl_scale']*kl_scale * kl_loss_ii_prescale


  #worm specific logvar for log_likelihood
  logvar = lax.dynamic_slice(params['logvar'], [worm,0,0],[1,data_dim] )[0]
  
  # Log-likelihood of data given latents nl_RNN
  out_mean_bxt = lfads['out_mean_t']
  out_nl_reg = lfads_hps['out_nl_reg']
  log_p_xgz = out_nl_reg*np.sum(dists.diag_gaussian_log_likelihood(x_bxt,
                                       mean= out_mean_bxt,
                                       logvar=logvar,
                                       varmin=lfads_hps['var_min'])) / B

  # Log-likelihood of data given latents approx_rnn
  out_mean_approx_bxt = lfads['out_mean_approx_t']
  out_staylor_reg = lfads_hps['out_staylor_reg']
  log_p_approx_xgz = out_staylor_reg*np.sum(
    dists.diag_gaussian_log_likelihood(x_bxt,
                                       mean= out_mean_approx_bxt,
                                       logvar=logvar,
                                       varmin=lfads_hps['var_min'] )) / B


  

  
  #Fixed point regularization
  fp_reg = lfads_hps['fp_reg']
  F_star_t = lfads['F_star_t']
  g_star_t = lfads['g_star_t']
  fp_loss = fp_reg*np.sum((F_star_t-g_star_t)**2)/B

  #Taylor regularization
  taylor_reg = lfads_hps['taylor_reg']
  gen_t = lfads['gen_t']
  g_approx_t = lfads['g_approx_t']
  taylor_loss = taylor_reg*np.sum((gen_t-g_approx_t)**2)/B

  # L2
  l2reg = lfads_hps['l2reg']
  l2_loss = l2reg * optimizers.l2_norm(params)**2

  loss = -log_p_xgz - log_p_approx_xgz +\
    kl_loss_g0 + kl_loss_ii + l2_loss + fp_loss + taylor_loss
  all_losses = {'total' : loss, 'nlog_p_xgz' : -log_p_xgz,
                'nlog_p_approx_xgz' : -log_p_approx_xgz,
                'kl_g0' : kl_loss_g0, 'kl_g0_prescale' : kl_loss_g0_prescale,
                'kl_ii' : kl_loss_ii, 'kl_ii_prescale' : kl_loss_ii_prescale,
                'l2' : l2_loss, 'fp_loss': fp_loss, 'taylor_loss': taylor_loss}
  return all_losses


def lfads_training_loss(params, lfads_hps, key, x_bxt, kl_scale, keep_rate, worm):
  """Pull out the total loss for training.

  Arguments:
    params: a dictionary of LFADS parameters
    lfads_hps: a dictionary of LFADS hyperparameters
    key: random.PRNGKey for random bits
    x_bxt: np array of input with leading dims being batch and time
    keep_rate: dropout keep rate
    kl_scale: scale on KL

  Returns:
    return the total loss for optimization
  """
  losses = lfads_losses(params, lfads_hps, key, x_bxt, kl_scale, keep_rate, worm)
  return losses['total']


def posterior_sample_and_average(params, lfads_hps, key, x_txd, worm):
  """Get the denoised lfad inferred values by posterior sample and average.

  Arguments:
    params: dictionary of lfads parameters
    lfads_hps: dict of LFADS hyperparameters
    key: JAX random state
    x_txd: 2d np.array time by dim trial to denoise

  Returns: 
    LFADS dictionary of inferred values, averaged over randomness.
  """
  batch_size = lfads_hps['batch_size']
  skeys = random.split(key, batch_size)  
  x_bxtxd = np.repeat(np.expand_dims(x_txd, axis=0), batch_size, axis=0)
  keep_rate = 1.0
  lfads_dict = batch_lfads(params, lfads_hps, skeys, x_bxtxd, keep_rate, worm)
  return utils.average_lfads_batch(lfads_dict)


### JIT

# JIT functions are orders of magnitude faster.  The first time you use them,
# they will take a couple of minutes to compile, then the second time you use
# them, they will be blindingly fast.

# The static_argnums is telling JAX to ignore the lfads_hps dictionary,
# which means you'll have to pay attention if you change the params.
# How does one force a recompile?


batch_lfads_jit = jit(batch_lfads, static_argnums=(1,))
lfads_losses_jit = jit(lfads_losses, static_argnums=(1,))
lfads_training_loss_jit = jit(lfads_training_loss, static_argnums=(1,))
posterior_sample_and_average_jit = jit(posterior_sample_and_average, static_argnums=(1,))


#batch_lfads_jit = jit(batch_lfads)
#lfads_losses_jit = jit(lfads_losses)
#lfads_training_loss_jit = jit(lfads_training_loss)
#posterior_sample_and_average_jit = jit(posterior_sample_and_average)

                  
  
