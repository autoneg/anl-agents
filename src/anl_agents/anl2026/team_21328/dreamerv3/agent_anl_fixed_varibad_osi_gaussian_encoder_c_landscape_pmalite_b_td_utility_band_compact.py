import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from .anl_components import FixedCandidateUtilityBandPolicyHead
from .anl_components import FixedProposalActionEncoder
from .anl_components import fixed_5x6_proposals
from .anl_components import proposal_mask_from_obs
from .anl_components import proposal_utility_stats
from .anl_components import score_proposals_from_table
from .anl_components import score_offer_from_table

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
pred = lambda xs: jax.tree.map(lambda x: x.pred(), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3

TARGET_ACTION_KEYS = (
    'target_z_e',
    'target_z_delta_down',
    'target_z_delta_up',
    'target_z_umax',
    'target_rho',
    'target_delta_down',
    'target_delta_up',
    'target_u_max',
)


def _safe_logit(x):
  x = f32(x)
  return jnp.log(x) - jnp.log1p(-x)


class GaussianOSIOutput:

  def __init__(self, mean_cont, log_std_cont, pref_logits, pref_scale=1.0):
    self.mean_cont = f32(mean_cont)
    self.log_std_cont = jnp.clip(f32(log_std_cont), -5.0, 2.0)
    self.pref_logits = f32(pref_logits)
    self.pref_scale = pref_scale

  def pred(self):
    pref = jax.nn.sigmoid(self.pref_logits)
    return jnp.concatenate([
        self.mean_cont[..., :3],
        pref,
        self.mean_cont[..., 3:],
    ], -1)

  def std(self):
    std_cont = jnp.exp(self.log_std_cont)
    pref = jax.nn.sigmoid(self.pref_logits)
    pref_std = jnp.sqrt(jnp.maximum(pref * (1.0 - pref), 1e-6))
    return jnp.concatenate([
        std_cont[..., :3],
        pref_std,
        std_cont[..., 3:],
    ], -1)

  def loss(self, target):
    cont_target = jnp.concatenate([target[..., :3], target[..., 6:8]], -1)
    pref_target = target[..., 3:6]
    var = jnp.exp(2.0 * self.log_std_cont)
    cont = 0.5 * (
        jnp.square(cont_target - self.mean_cont) / var +
        2.0 * self.log_std_cont + jnp.log(2.0 * jnp.pi))
    pref = (
        jax.nn.softplus(self.pref_logits) -
        f32(pref_target) * self.pref_logits)
    return cont.mean(-1) + self.pref_scale * pref.mean(-1)

  def metrics(self, target):
    cont_target = jnp.concatenate([target[..., :3], target[..., 6:8]], -1)
    pref_target = target[..., 3:6]
    var = jnp.exp(2.0 * self.log_std_cont)
    cont = 0.5 * (
        jnp.square(cont_target - self.mean_cont) / var +
        2.0 * self.log_std_cont + jnp.log(2.0 * jnp.pi))
    pref = (
        jax.nn.softplus(self.pref_logits) -
        f32(pref_target) * self.pref_logits)
    std = self.std()
    return {
        'osi/nll': self.loss(target).mean(),
        'osi/cont_nll': cont.mean(),
        'osi/pref_bce': pref.mean(),
        'osi/std_mean': std.mean(),
        'osi/std_weights': std[..., :3].mean(),
        'osi/std_reservation': std[..., 6].mean(),
        'osi/std_discount': std[..., 7].mean(),
    }


class GaussianOSIHead(nj.Module):

  def __init__(self, theta_dim, layers=2, units=128, act='silu', norm='rms',
      output='hybrid_gaussian', outscale=1.0, winit='trunc_normal_in',
      binit='zeros', log_std_min=-5.0, log_std_max=2.0, pref_scale=1.0):
    assert theta_dim == 8, theta_dim
    assert output == 'hybrid_gaussian', output
    self.theta_dim = theta_dim
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.outscale = outscale
    self.winit = winit
    self.binit = binit
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.pref_scale = pref_scale

  def __call__(self, x, bdims):
    bshape = jax.tree.leaves(x)[0].shape[:bdims]
    x = x.reshape((*bshape, -1))
    x = self.sub(
        'mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm, winit=self.winit, binit=self.binit)(x)
    kw = dict(winit=self.winit, binit=self.binit, outscale=self.outscale)
    mean_cont = self.sub('mean_cont', nn.Linear, 5, **kw)(x)
    log_std_cont = self.sub('log_std_cont', nn.Linear, 5, **kw)(x)
    log_std_cont = jnp.clip(log_std_cont, self.log_std_min, self.log_std_max)
    pref_logits = self.sub('pref_logits', nn.Linear, 3, **kw)(x)
    return GaussianOSIOutput(
        mean_cont, log_std_cont, pref_logits, self.pref_scale)


class FrequencyResidualTableOSIOutput:

  def __init__(
      self, freq, weight_logits, value_logits, reservation_mean, logstd_w, logstd_v,
      reservation_logstd,
      type_logits, num_tasks, issue_mask, value_mask, relative_time,
      minstd=0.03):
    self.freq = freq
    self.weight_logits = f32(weight_logits)
    self.value_logits = f32(value_logits)
    self.reservation_mean_raw = f32(reservation_mean)
    self.logstd_w = jnp.clip(f32(logstd_w), -5.0, 2.0)
    self.logstd_v = jnp.clip(f32(logstd_v), -5.0, 2.0)
    self.reservation_logstd = jnp.clip(f32(reservation_logstd), -5.0, 2.0)
    self.type_logits = f32(type_logits)
    self.num_tasks = int(num_tasks)
    self.issue_mask = issue_mask.astype(bool)
    self.value_mask = value_mask.astype(bool)
    self.relative_time = jnp.clip(f32(relative_time), 0.0, 1.0)
    self.minstd = float(minstd)

  def _confidence(self, std):
    std = sg(std)
    gate = jnp.clip(self.relative_time / f32(0.1), 0.0, 1.0)
    while gate.ndim < std.ndim:
      gate = gate[..., None]
    base = jnp.minimum(jnp.exp(-f32(6.0) * std), f32(0.85))
    return gate * base

  def _raw_weights_mean(self):
    logits = self.weight_logits
    logits = jnp.where(self.issue_mask, logits, -1e30)
    return jax.nn.softmax(logits, -1).astype(f32)

  def _raw_values_mean(self):
    mean = jax.nn.sigmoid(self.value_logits).astype(f32)
    return jnp.where(self.value_mask, mean, 0.0)

  def _raw_reservation_mean(self):
    return jnp.clip(self.reservation_mean_raw, 0.0, 0.6).astype(f32)

  def _raw_weights_std(self):
    std = jax.nn.softplus(self.logstd_w) + self.minstd
    return jnp.where(self.issue_mask, std, 0.0).astype(f32)

  def _raw_values_std(self):
    std = jax.nn.softplus(self.logstd_v) + self.minstd
    return jnp.where(self.value_mask, std, 0.0).astype(f32)

  def _raw_reservation_std(self):
    return (jax.nn.softplus(self.reservation_logstd) + self.minstd).astype(f32)

  def weights_mean(self):
    raw_mean = self._raw_weights_mean()
    raw_std = self._raw_weights_std()
    freq_mean = self.freq['weights_mean']
    confidence = self._confidence(raw_std)
    mixed = confidence * raw_mean + (1.0 - confidence) * freq_mean
    mixed = jnp.where(self.issue_mask, jnp.maximum(mixed, 0.0), 0.0)
    denom = jnp.maximum(mixed.sum(-1, keepdims=True), f32(1e-8))
    return (mixed / denom).astype(f32)

  def values_mean(self):
    raw_mean = self._raw_values_mean()
    raw_std = self._raw_values_std()
    freq_mean = self.freq['values_mean']
    confidence = self._confidence(raw_std)
    mixed = confidence * raw_mean + (1.0 - confidence) * freq_mean
    mixed = jnp.clip(mixed, 0.0, 1.0)
    return jnp.where(self.value_mask, mixed, 0.0).astype(f32)

  def reservation_mean(self):
    return jnp.clip(self._raw_reservation_mean(), 0.0, 1.0).astype(f32)

  def weights_std(self):
    raw_std = self._raw_weights_std()
    freq_std = self.freq['weights_std']
    confidence = self._confidence(raw_std)
    mixed = jnp.sqrt(
        jnp.square(confidence * raw_std) +
        jnp.square((1.0 - confidence) * freq_std))
    return jnp.where(self.issue_mask, mixed, 0.0).astype(f32)

  def values_std(self):
    raw_std = self._raw_values_std()
    freq_std = self.freq['values_std']
    confidence = self._confidence(raw_std)
    mixed = jnp.sqrt(
        jnp.square(confidence * raw_std) +
        jnp.square((1.0 - confidence) * freq_std))
    return jnp.where(self.value_mask, mixed, 0.0).astype(f32)

  def reservation_std(self):
    return self._raw_reservation_std()

  def pred(self):
    return {
        'weights_mean': self.weights_mean(),
        'weights_std': self.weights_std(),
        'values_mean': self.values_mean(),
        'values_std': self.values_std(),
        'raw_weights_mean': self._raw_weights_mean(),
        'raw_weights_std': self._raw_weights_std(),
        'raw_values_mean': self._raw_values_mean(),
        'raw_values_std': self._raw_values_std(),
        'reservation_mean': self.reservation_mean(),
        'reservation_std': self.reservation_std(),
        'type_probs': jax.nn.softmax(self.type_logits, -1).astype(f32),
        'type_logits': self.type_logits,
    }

  def _nll(self, target, mean, std):
    var = jnp.square(jnp.maximum(std, self.minstd))
    return 0.5 * (
        jnp.square(target - mean) / var + jnp.log(var) + jnp.log(2.0 * jnp.pi))

  def loss(self, obs):
    pred = self.pred()
    issue_mask = self.issue_mask.astype(f32)
    value_mask = self.value_mask.astype(f32)
    raw_weight_nll = self._nll(
        obs['true_opp_weights'],
        pred['raw_weights_mean'], pred['raw_weights_std'])
    raw_value_nll = self._nll(
        obs['true_opp_values'],
        pred['raw_values_mean'], pred['raw_values_std'])
    mixed_weight_nll = self._nll(
        obs['true_opp_weights'], pred['weights_mean'], pred['weights_std'])
    mixed_value_nll = self._nll(
        obs['true_opp_values'], pred['values_mean'], pred['values_std'])
    reservation_nll = self._nll(
        obs['true_opp_reserved'][..., None],
        pred['reservation_mean'], pred['reservation_std']).squeeze(-1)
    raw_weight_loss = (
        (raw_weight_nll * issue_mask).sum() /
        jnp.maximum(f32(1.0), issue_mask.sum()))
    raw_value_loss = (
        (raw_value_nll * value_mask).sum() /
        jnp.maximum(f32(1.0), value_mask.sum()))
    mixed_weight_loss = (
        (mixed_weight_nll * issue_mask).sum() /
        jnp.maximum(f32(1.0), issue_mask.sum()))
    mixed_value_loss = (
        (mixed_value_nll * value_mask).sum() /
        jnp.maximum(f32(1.0), value_mask.sum()))
    task_target = obs['opponent_type_id']
    task_onehot = jax.nn.one_hot(task_target, self.num_tasks, dtype=f32)
    task_logprob = jax.nn.log_softmax(self.type_logits, -1)
    type_loss = -(task_onehot * task_logprob).sum(-1).mean()
    raw_loss = raw_weight_loss + raw_value_loss
    mixed_loss = mixed_weight_loss + mixed_value_loss
    return (
        raw_loss + f32(0.2) * mixed_loss + reservation_nll.mean() + type_loss)

  def metrics(self, obs):
    pred = self.pred()
    issue_mask = self.issue_mask.astype(f32)
    value_mask = self.value_mask.astype(f32)
    weight_mae = (
        (jnp.abs(pred['weights_mean'] - obs['true_opp_weights']) * issue_mask).sum() /
        jnp.maximum(f32(1.0), issue_mask.sum()))
    value_mae = (
        (jnp.abs(pred['values_mean'] - obs['true_opp_values']) * value_mask).sum() /
        jnp.maximum(f32(1.0), value_mask.sum()))
    raw_weight_mae = (
        (jnp.abs(pred['raw_weights_mean'] - obs['true_opp_weights']) * issue_mask).sum() /
        jnp.maximum(f32(1.0), issue_mask.sum()))
    raw_value_mae = (
        (jnp.abs(pred['raw_values_mean'] - obs['true_opp_values']) * value_mask).sum() /
        jnp.maximum(f32(1.0), value_mask.sum()))
    reservation_mae = jnp.abs(
        pred['reservation_mean'].squeeze(-1) - obs['true_opp_reserved']).mean()
    task_target = obs['opponent_type_id']
    task_pred = jnp.argmax(self.type_logits, -1)
    return {
        'osi_table/nll': self.loss(obs),
        'osi_table/weight_mae': weight_mae,
        'osi_table/value_mae': value_mae,
        'osi_table/raw_weight_mae': raw_weight_mae,
        'osi_table/raw_value_mae': raw_value_mae,
        'osi_table/reservation_mae': reservation_mae,
        'osi_table/weight_std': (
            (pred['weights_std'] * issue_mask).sum() /
            jnp.maximum(f32(1.0), issue_mask.sum())),
        'osi_table/value_std': (
            (pred['values_std'] * value_mask).sum() /
            jnp.maximum(f32(1.0), value_mask.sum())),
        'osi_table/reservation_std': pred['reservation_std'].mean(),
        'osi_table/type_acc': (task_pred == task_target).mean(),
    }


class FrequencyResidualTableOSIHead(nj.Module):

  def __init__(
      self, max_issues=6, max_values=5, layers=2, units=128, act='silu',
      norm='rms', minstd=0.03, outscale=1.0, winit='trunc_normal_in',
      binit='zeros', alpha_max=0.35, alpha_tau=5.0, sigma_ref=0.35,
      max_delta=1.0, num_tasks=25, output='gaussian_table'):
    assert output == 'gaussian_table', output
    self.max_issues = max_issues
    self.max_values = max_values
    self.num_tasks = int(num_tasks)
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.minstd = minstd
    self.outscale = outscale
    self.winit = winit
    self.binit = binit
    self.alpha_max = float(alpha_max)
    self.alpha_tau = float(alpha_tau)
    self.sigma_ref = float(sigma_ref)

  def __call__(self, feat, obs, bdims):
    bshape = feat.shape[:bdims]
    x = feat.reshape((*bshape, -1))
    opp_offer_freq = self._counts_frequency(obs, 'opp_offer_counts')
    self_offer_freq = self._counts_frequency(obs, 'self_offer_counts')
    x = jnp.concatenate([
        x,
        obs['relative_time'][..., None],
        opp_offer_freq.reshape((*bshape, -1)),
        self_offer_freq.reshape((*bshape, -1)),
    ], -1)
    x = self.sub(
        'residual_mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(x)
    kw = dict(winit=self.winit, binit=self.binit, outscale=self.outscale)
    weight_logits = self.sub('weight_logits', nn.Linear, self.max_issues, **kw)(x)
    value_logits = self.sub(
        'value_logits', nn.Linear, (self.max_issues, self.max_values), **kw)(x)
    reservation_mean = self.sub('reservation_mean', nn.Linear, 1, **kw)(x)
    logstd_w = self.sub('residual_weight_logstd', nn.Linear, self.max_issues, **kw)(x)
    logstd_v = self.sub(
        'residual_value_logstd', nn.Linear,
        (self.max_issues, self.max_values), **kw)(x)
    reservation_logstd = self.sub('reservation_logstd', nn.Linear, 1, **kw)(x)
    type_logits = self.sub(
        'type_logits', nn.Linear, self.num_tasks, **kw)(x)
    freq = self._frequency_belief(obs)
    return FrequencyResidualTableOSIOutput(
        freq, weight_logits, value_logits, reservation_mean, logstd_w, logstd_v,
        reservation_logstd,
        type_logits, self.num_tasks, obs['issue_mask'], obs['value_mask'],
        obs['relative_time'], self.minstd)

  def _frequency_belief(self, obs):
    issue_mask = obs['issue_mask'].astype(f32)
    value_mask = obs['value_mask'].astype(f32)
    counts = jnp.maximum(f32(obs['opp_offer_counts']), 0.0) * value_mask
    totals = jnp.maximum(counts.sum(-1, keepdims=True), f32(1.0))
    probs = counts / totals
    values = jnp.where(value_mask > 0, probs, 0.0)
    spread = jnp.sqrt(
        jnp.maximum(
            ((probs - (probs * value_mask).sum(-1, keepdims=True) /
              jnp.maximum(value_mask.sum(-1, keepdims=True), f32(1.0))) ** 2 *
             value_mask).sum(-1) /
            jnp.maximum(value_mask.sum(-1), f32(1.0)),
            f32(0.0)))
    spread = jnp.where(issue_mask > 0, spread, 0.0)
    denom = spread.sum(-1, keepdims=True)
    uniform = issue_mask / jnp.maximum(issue_mask.sum(-1, keepdims=True), f32(1.0))
    weights = jnp.where(denom > 1e-8, spread / denom, uniform)
    observed = jnp.maximum(counts - value_mask, 0.0)
    active_issues = jnp.maximum(issue_mask.sum(-1), f32(1.0))
    nobs = (observed.sum(-1) * issue_mask).sum(-1) / active_issues
    sigma_scalar = 1.0 / jnp.sqrt(nobs + 1.0)
    weights_std = (
        f32(0.35) * sigma_scalar[..., None] *
        issue_mask / jnp.maximum(issue_mask.sum(-1, keepdims=True), f32(1.0)))
    values_std = f32(0.5) * sigma_scalar[..., None, None] * value_mask
    return {
        'weights_mean': weights.astype(f32),
        'weights_std': weights_std.astype(f32),
        'values_mean': values.astype(f32),
        'values_std': values_std.astype(f32),
        'nobs': nobs.astype(f32),
    }

  def _counts_frequency(self, obs, key):
    value_mask = obs['value_mask'].astype(f32)
    counts = jnp.maximum(f32(obs[key]), 0.0)
    counts = counts * value_mask
    denom = jnp.maximum(counts.sum(-1, keepdims=True), f32(1.0))
    return (counts / denom).astype(f32)

class BranchedHeadFusion(nj.Module):

  def __init__(self, feat_units=(256, 128), global_units=(128, 64),
      deter_dim=None, stoch_units=None, act='silu',
      winit='trunc_normal_in', binit='zeros'):
    self.feat_units = tuple(feat_units)
    self.global_units = tuple(global_units)
    self.deter_dim = int(deter_dim or 0)
    self.stoch_units = int(stoch_units or 0)
    self.act = act
    self.winit = winit
    self.binit = binit

  def __call__(self, feat_tensor, global_extra, bdims, task_z=None):
    bshape = feat_tensor.shape[:bdims]
    feat_tensor = nn.cast(feat_tensor.reshape((*bshape, -1)))
    global_extra = nn.cast(global_extra.reshape((*bshape, -1)))
    h_feat = self._feature_branch(feat_tensor)
    h_global = self._stack('global', global_extra, self.global_units)
    parts = [h_feat, h_global]
    if task_z is not None:
      parts.append(nn.cast(task_z.reshape((*bshape, -1))))
    return jnp.concatenate(parts, -1)

  def _feature_branch(self, feat_tensor):
    if self.deter_dim <= 0 or self.stoch_units <= 0:
      return self._stack('feat', feat_tensor, self.feat_units)
    deter = feat_tensor[..., :self.deter_dim]
    stoch = feat_tensor[..., self.deter_dim:]
    stoch = self.sub(
        'stoch_0', nn.Linear, self.stoch_units,
        winit=self.winit, binit=self.binit)(stoch)
    stoch = getattr(jax.nn, self.act)(stoch)
    return self._stack(
        'feat', jnp.concatenate([deter, stoch], -1), self.feat_units)

  def _stack(self, prefix, x, units_list):
    for i, units in enumerate(units_list):
      x = self.sub(
          f'{prefix}_{i}', nn.Linear, units,
          winit=self.winit, binit=self.binit)(x)
      x = getattr(jax.nn, self.act)(x)
    return x


class TaskBeliefOutput:

  def __init__(self, mean, logvar, sample):
    self.mean = f32(mean)
    self.logvar = jnp.clip(f32(logvar), -10.0, 5.0)
    self.z = f32(sample)

  def kl_standard_normal(self):
    return -0.5 * (
        1.0 + self.logvar - jnp.square(self.mean) - jnp.exp(self.logvar)
    ).sum(-1)


class TaskBeliefHead(nj.Module):

  def __init__(self, latent_dim=16, layers=2, units=128, act='silu',
      norm='rms', winit='trunc_normal_in', binit='zeros', outscale=1.0):
    self.latent_dim = latent_dim
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit
    self.outscale = outscale

  def __call__(self, feat_tensor, bdims, sample=True):
    bshape = feat_tensor.shape[:bdims]
    x = nn.cast(feat_tensor.reshape((*bshape, -1)))
    x = self.sub(
        'mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm, winit=self.winit, binit=self.binit)(x)
    kw = dict(winit=self.winit, binit=self.binit, outscale=self.outscale)
    mean = self.sub('mean', nn.Linear, self.latent_dim, **kw)(x)
    logvar = self.sub('logvar', nn.Linear, self.latent_dim, **kw)(x)
    logvar = jnp.clip(logvar, -10.0, 5.0)
    if sample:
      eps = jax.random.normal(nj.seed(), mean.shape)
      z = mean + jnp.exp(0.5 * logvar) * eps
    else:
      z = mean
    return TaskBeliefOutput(mean, logvar, z)


class VaribadAdapter(nj.Module):

  def __init__(self, output_dim, layers=(128,), act='silu',
      winit='trunc_normal_in', binit='zeros', outscale=1.0):
    self.output_dim = output_dim
    self.layers = tuple(layers)
    self.act = act
    self.winit = winit
    self.binit = binit
    self.outscale = outscale

  def __call__(self, x, bdims):
    bshape = x.shape[:bdims]
    x = nn.cast(x.reshape((*bshape, -1)))
    for i, units in enumerate(self.layers):
      x = self.sub(
          f'hidden_{i}', nn.Linear, units,
          winit=self.winit, binit=self.binit)(x)
      x = getattr(jax.nn, self.act)(x)
    return self.sub(
        'out', nn.Linear, self.output_dim, winit=self.winit,
        binit=self.binit, outscale=self.outscale)(x)


class UtilityBandOutput(outs.Output):

  def __init__(
      self, action_logits, rho, delta, z_rho, z_delta, mu_rho, logstd_rho,
      mu_delta, logstd_delta, z_umax, mu_umax, logstd_umax, reservation,
      band_logits, u_max, det_rho, det_delta, det_u_max, det_logits,
      u_max_min=0.9, u_max_max=1.0, unimix=0.01):
    self.action = outs.Categorical(action_logits, unimix)
    self.det_action = outs.Categorical(det_logits, unimix)
    self.rho = f32(rho)
    self.delta = f32(delta)
    self.z_rho = f32(z_rho)
    self.z_delta = f32(z_delta)
    self.z_umax = f32(z_umax)
    self.mu_rho = f32(mu_rho)
    self.logstd_rho = jnp.clip(f32(logstd_rho), -5.0, 2.0)
    self.mu_delta = f32(mu_delta)
    self.logstd_delta = jnp.clip(f32(logstd_delta), -5.0, 2.0)
    self.mu_umax = f32(mu_umax)
    self.logstd_umax = jnp.clip(f32(logstd_umax), -5.0, 2.0)
    self.reservation = f32(reservation)
    self.band_logits = f32(band_logits)
    self.u_max = f32(u_max)
    self.det_rho = f32(det_rho)
    self.det_delta = f32(det_delta)
    self.det_u_max = f32(det_u_max)
    self.u_max_min = float(u_max_min)
    self.u_max_max = float(u_max_max)
    self.minent = 0
    self.maxent = np.log(action_logits.shape[-1])

  def pred(self):
    return self.det_action.pred()

  def sample(self, seed, shape=()):
    return self.action.sample(seed, shape)

  def logp(self, event):
    return self._logp_band() + self.action.logp(event)

  def entropy(self):
    rho_std = jnp.exp(self.logstd_rho)
    delta_std = jnp.exp(self.logstd_delta)
    umax_std = jnp.exp(self.logstd_umax)
    eps = f32(1e-6)
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    band_ent = (
        0.5 * jnp.log(2 * jnp.pi * jnp.square(rho_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(delta_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(umax_std)) + 0.5 +
        logdet_umax)
    return band_ent.squeeze(-1) + self.action.entropy()

  def kl(self, other):
    raise NotImplementedError(type(other))

  def _logp_band(self):
    eps = f32(1e-6)
    e_std = jnp.exp(self.logstd_rho)
    delta_std = jnp.exp(self.logstd_delta)
    umax_std = jnp.exp(self.logstd_umax)
    logp_e = -0.5 * (
        jnp.square((self.z_rho - self.mu_rho) / e_std) +
        2.0 * self.logstd_rho + jnp.log(2.0 * jnp.pi))
    logp_delta = -0.5 * (
        jnp.square((self.z_delta - self.mu_delta) / delta_std) +
        2.0 * self.logstd_delta + jnp.log(2.0 * jnp.pi))
    logp_umax = -0.5 * (
        jnp.square((self.z_umax - self.mu_umax) / umax_std) +
        2.0 * self.logstd_umax + jnp.log(2.0 * jnp.pi))
    log_e_min = f32(np.log(2.0))
    log_e_max = f32(np.log(50.0))
    log_e_range = log_e_max - log_e_min
    eta = log_e_min + jax.nn.sigmoid(self.z_rho) * log_e_range
    logdet_e = (
        eta + jnp.log(jnp.maximum(log_e_range, eps)) +
        jax.nn.log_sigmoid(self.z_rho) +
        jax.nn.log_sigmoid(-self.z_rho))
    logdet_delta = (
        jnp.log(jnp.maximum(1.0 - sg(self.rho), eps)) +
        jax.nn.log_sigmoid(self.z_delta) +
        jax.nn.log_sigmoid(-self.z_delta))
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    return (
        logp_e - logdet_e +
        logp_delta - logdet_delta +
        logp_umax - logdet_umax).squeeze(-1)


class UtilityBandPolicyHead(nj.Module):

  def __init__(self, act_space, outputs, layers=3, units=1024, act='silu',
      norm='rms', minstd=0.1, maxstd=1.0, outscale=0.01, unimix=0.01,
      winit='trunc_normal_in', binit='zeros', e_min=2.0, e_max=50.0,
      rho_delay_coef=0.0, rho_delay_power=1.0, offer_delay_coef=0.0,
      accept_delay_coef=0.0, delay_power=2.0, u_max_min=0.9,
      u_max_max=1.0):
    assert set(act_space.keys()) == {'action'}, act_space
    assert outputs['action'] == 'categorical', outputs
    self.act_space = act_space
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.minstd = minstd
    self.maxstd = maxstd
    self.outscale = outscale
    self.unimix = unimix
    self.winit = winit
    self.binit = binit
    self.e_min = e_min
    self.e_max = e_max
    self.u_max_min = float(u_max_min)
    self.u_max_max = float(u_max_max)
    self.rho_delay_coef = rho_delay_coef
    self.rho_delay_power = rho_delay_power
    self.offer_delay_coef = offer_delay_coef
    self.accept_delay_coef = accept_delay_coef
    self.delay_power = delay_power

  def __call__(self, parts, bdims):
    bshape = parts['head'].shape[:bdims]
    head = nn.cast(parts['head'].reshape((*bshape, -1)))
    obs_vec = f32(parts['obs'].reshape((*bshape, -1)))
    band = self.sub(
        'band_mlp', nn.MLP, 2, 512,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(head)
    mu_e = self.sub(
        'mu_e', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_e = self.sub(
        'logstd_e', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    mu_delta = self.sub(
        'mu_delta', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_delta = self.sub(
        'logstd_delta', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    mu_umax = self.sub(
        'mu_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_umax = self.sub(
        'logstd_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_e = jnp.clip(logstd_e, -5.0, 2.0)
    logstd_delta = jnp.clip(logstd_delta, -5.0, 2.0)
    logstd_umax = jnp.clip(logstd_umax, -5.0, 2.0)
    e_std = jnp.exp(logstd_e)
    delta_std = jnp.exp(logstd_delta)
    umax_std = jnp.exp(logstd_umax)
    key_e, key_delta, key_umax = jax.random.split(nj.seed(), 3)
    z_e = mu_e + e_std * jax.random.normal(key_e, mu_e.shape)
    z_delta = mu_delta + delta_std * jax.random.normal(
        key_delta, mu_delta.shape)
    z_umax = mu_umax + umax_std * jax.random.normal(
        key_umax, mu_umax.shape)
    reservation = jnp.clip(obs_vec[..., 9:10], 0.0, 1.0)
    e = self._concession_e(z_e)
    det_e = self._concession_e(mu_e)
    time_ratio = jnp.clip(obs_vec[..., 0:1], 1e-6, 1.0)
    u_max = self._u_max(z_umax)
    det_u_max = self._u_max(mu_umax)
    rho = reservation + (u_max - reservation) * (
        1.0 - jnp.power(time_ratio, e))
    det_rho = reservation + (det_u_max - reservation) * (
        1.0 - jnp.power(time_ratio, det_e))
    rho = jnp.clip(rho, 0.0, 1.0)
    det_rho = jnp.clip(det_rho, 0.0, 1.0)
    delta = (1.0 - rho) * jax.nn.sigmoid(z_delta)
    det_delta = (1.0 - det_rho) * jax.nn.sigmoid(mu_delta)
    logits = self._action_logits(head, obs_vec, rho, delta)
    det_logits = self._action_logits(head, obs_vec, det_rho, det_delta)
    return {'action': UtilityBandOutput(
        logits, rho, delta, z_e, z_delta, mu_e, logstd_e, mu_delta,
        logstd_delta, z_umax, mu_umax, logstd_umax, reservation, band,
        u_max, det_rho, det_delta, det_u_max, det_logits,
        self.u_max_min, self.u_max_max, self.unimix)}

  def _concession_e(self, z_e):
    log_e_min = f32(np.log(self.e_min))
    log_e_max = f32(np.log(self.e_max))
    eta = log_e_min + jax.nn.sigmoid(z_e) * (log_e_max - log_e_min)
    return jnp.exp(eta)

  def _u_max(self, z_umax):
    return self.u_max_min + (
        self.u_max_max - self.u_max_min) * jax.nn.sigmoid(z_umax)

  def _action_logits(self, head, obs_vec, rho, delta):
    action_inp = jnp.concatenate([head, rho, delta], -1)
    x = self.sub(
        'action_mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(action_inp)
    classes = np.asarray(self.act_space['action'].classes).flatten()
    assert (classes == classes[0]).all(), classes
    logits = self.sub(
        'action_logits', nn.Linear, int(classes[0]),
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(x)
    return self._mask_logits(logits, obs_vec, rho, delta)

  def _mask_logits(self, logits, obs_vec, rho, delta):
    n_actions = logits.shape[-1]
    n_offers = n_actions - 1
    my_u = self._all_my_discounted_utilities(obs_vec)
    offer_mask = jnp.logical_and(my_u >= rho, my_u <= rho + delta)
    nearest = jnp.argmin(jnp.abs(my_u - rho), -1)
    fallback = jax.nn.one_hot(nearest, n_offers).astype(bool)
    has_offer = offer_mask.any(-1, keepdims=True)
    offer_mask = jnp.where(has_offer, offer_mask, fallback)
    accept_allowed = self._last_opp_discounted_utility(obs_vec) >= rho
    mask = jnp.concatenate([offer_mask, accept_allowed], -1)
    return jnp.where(mask, logits, jnp.full_like(logits, -1e9))

  def _all_my_discounted_utilities(self, obs_vec):
    time = obs_vec[..., 0:1]
    my_discount = jnp.clip(obs_vec[..., 10:11], 1e-4, 1.0)
    my_weights = obs_vec[..., 11:14]
    my_pref = obs_vec[..., 14:17]
    proposals = jnp.asarray(self._proposals(), f32)
    offers = proposals.reshape((*(1,) * (obs_vec.ndim - 1), *proposals.shape))
    offers = offers + jnp.zeros((*obs_vec.shape[:-1], *proposals.shape), f32)
    values = my_pref[..., None, :] * offers + (
        1.0 - my_pref[..., None, :]) * (1.0 - offers)
    raw = (my_weights[..., None, :] * values).sum(-1)
    return raw * jnp.power(my_discount, time)

  def _last_opp_discounted_utility(self, obs_vec):
    time = obs_vec[..., 0:1]
    last_opp = obs_vec[..., 1:4]
    valid = (last_opp >= 0.0).all(-1, keepdims=True)
    offer = jnp.where(valid, last_opp, jnp.zeros_like(last_opp))
    my_discount = jnp.clip(obs_vec[..., 10:11], 1e-4, 1.0)
    my_weights = obs_vec[..., 11:14]
    my_pref = obs_vec[..., 14:17]
    values = my_pref * offer + (1.0 - my_pref) * (1.0 - offer)
    raw = (my_weights * values).sum(-1, keepdims=True)
    return jnp.where(valid, raw * jnp.power(my_discount, time), -jnp.inf)

  def _proposals(self):
    n_actions = int(np.asarray(self.act_space['action'].classes).flatten()[0])
    n_proposals = n_actions - 1
    n_issues = 3
    issue_size = round(n_proposals ** (1.0 / n_issues))
    issue_sizes = [issue_size] * n_issues
    assert int(np.prod(issue_sizes)) == n_proposals
    outcomes = np.array(
        np.meshgrid(*[np.arange(s) for s in issue_sizes], indexing='ij')
    ).reshape(n_issues, -1).T
    return outcomes.astype(np.float32) / (
        np.asarray(issue_sizes, np.float32) - 1.0)


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.config = config
    self.max_issues = 6
    self.max_values = 5
    self.act_space = self._augment_act_space(act_space)
    self.proposal_semantic_dim = 2 * 6 * self.max_issues
    self.encoder_obs_dim = (
        27 + self.max_issues + self.max_issues * self.max_values +
        self.proposal_semantic_dim + 4)
    self.action_emb_units = config.action_context.units

    exclude = (
        'is_first', 'is_last', 'is_terminal', 'reward',
        'true_opp_weights', 'true_opp_values', 'true_opp_reserved',
        'opponent_type_id', 'proposal_mask', 'issue_mask', 'value_mask',
        'opp_offer_counts', 'self_offer_counts',
        'last_opp_valid', 'last_self_valid',
        'osi_global_offset_steps', 'osi_global_total_steps')
    obs_space = dict(obs_space)
    enc_space = {}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    enc_space['obs'] = elements.Space(
        np.float32, (self.encoder_obs_dim,), -np.inf, np.inf)
    obs_space['obs'] = enc_space['obs']
    dec_space['obs'] = enc_space['obs']
    self.obs_space = obs_space
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn_act_space = {
        'action_emb': elements.Space(
            np.float32, (self.action_emb_units,), -np.inf, np.inf)}
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](
        self.dyn_act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')
    osi_kw = dict(config.osi.head)
    osi_kw.pop('output', None)
    osi_kw.setdefault('num_tasks', config.varibad.num_tasks)
    self.osi = FrequencyResidualTableOSIHead(
        max_issues=self.max_issues, max_values=self.max_values,
        **osi_kw, name='osi')
    self.proposals = fixed_5x6_proposals()
    self.action_enc = FixedProposalActionEncoder(
        self.proposals,
        units=self.action_emb_units,
        layers=config.action_context.layers,
        act=config.policy.act,
        norm=config.policy.norm,
        winit=config.policy.winit,
        name='action_enc')
    self.fusion = BranchedHeadFusion(
        feat_units=config.head_fusion.feat_units,
        global_units=config.head_fusion.global_units,
        deter_dim=config.head_fusion.deter_dim,
        stoch_units=config.head_fusion.stoch_units,
        act=config.policy.act,
        winit=config.policy.winit,
        name='fusion')
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = FixedCandidateUtilityBandPolicyHead(
        self.proposals, **config.policy, name='pol')

    self.val_target = embodied.jax.MLPHead(
        scalar, **config.value, name='val_target')
    self.slowval_target = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval_target'),
        source=self.val_target, **config.slowvalue)
    self.val_offer = embodied.jax.MLPHead(
        scalar, **config.value, name='val_offer')
    self.slowval_offer = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval_offer'),
        source=self.val_offer, **config.slowvalue)
    self.val = self.val_target
    self.slowval = self.slowval_target

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')
    self.reward_target = getattr(config, 'reward_target', 'env')

    self.modules = [
        self.dyn, self.action_enc, self.enc, self.dec, self.rew, self.con,
        self.osi, self.fusion, self.pol, self.val_target, self.val_offer]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales

  def _augment_act_space(self, act_space):
    spaces = dict(act_space)
    spaces.setdefault(
        'dreamer_opp_weights',
        elements.Space(np.float32, (self.max_issues,), 0.0, 1.0))
    spaces.setdefault(
        'dreamer_opp_values',
        elements.Space(
            np.float32, (self.max_issues, self.max_values), 0.0, 1.0))
    for key in TARGET_ACTION_KEYS:
      spaces.setdefault(
          key, elements.Space(np.float32, (), -np.inf, np.inf))
    return spaces

  @property
  def policy_keys(self):
    return '^(enc|dyn|action_enc|dec|pol|osi|fusion)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self._osi_schedule() == 'env_steps':
      spaces['env_step'] = elements.Space(np.float32)
      spaces['total_env_steps'] = elements.Space(np.float32)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space),
        self._prior_belief((batch_size,)),
        jnp.zeros((batch_size,), f32))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def _osi_schedule(self):
    return getattr(self.config.osi, 'schedule', 'env_steps')

  def _osi_alpha(self, data=None):
    if self._osi_schedule() == 'oracle':
      return f32(0.0)
    if self._osi_schedule() == 'env_steps':
      assert data is not None, 'env_steps OSI schedule requires run context.'
      step = jnp.max(f32(data['env_step']))
      total = jnp.maximum(f32(1.0), jnp.max(f32(data['total_env_steps'])))
      if 'osi_global_total_steps' in data:
        global_total = jnp.max(f32(data['osi_global_total_steps']))
        global_offset = jnp.max(f32(data['osi_global_offset_steps']))
        step = jnp.where(global_total > 0.0, global_offset + step, step)
        total = jnp.where(global_total > 0.0, global_total, total)
    else:
      total = f32(max(1, int(self.config.osi.total_train_steps)))
      step = f32(self.opt.step.read())
    p = jnp.minimum(f32(1.0), step / total)
    start = f32(self.config.osi.warmup_frac)
    ramp = f32(max(1e-6, float(self.config.osi.ramp_frac)))
    return jnp.clip((p - start) / ramp, f32(0.0), f32(1.0))

  def _policy_alpha(self, policy_step, mode, obs=None):
    if self._osi_schedule() == 'oracle':
      return f32(0.0)
    if mode.startswith('eval'):
      return f32(1.0)
    if self._osi_schedule() == 'env_steps':
      envs = f32(max(1, int(self.config.run.envs)))
      total = f32(max(1.0, float(self.config.run.steps)))
      p = jnp.minimum(f32(1.0), f32(policy_step) * envs / total)
      if obs is not None and 'osi_global_total_steps' in obs:
        global_total = jnp.max(f32(obs['osi_global_total_steps']))
        global_offset = jnp.max(f32(obs['osi_global_offset_steps']))
        local_step = f32(policy_step) * envs
        p = jnp.where(
            global_total > 0.0,
            jnp.minimum(f32(1.0), (global_offset + local_step) / global_total),
            p)
    else:
      total = f32(max(1, int(self.config.osi.total_train_steps)))
      p = jnp.minimum(f32(1.0), f32(self.opt.step.read()) / total)
    start = f32(self.config.osi.warmup_frac)
    ramp = f32(max(1e-6, float(self.config.osi.ramp_frac)))
    return jnp.clip((p - start) / ramp, f32(0.0), f32(1.0))

  def _flatten_belief(self, belief):
    return jnp.concatenate([
        belief['weights_mean'],
        belief['weights_std'],
        belief['values_mean'].reshape((*belief['values_mean'].shape[:-2], -1)),
        belief['values_std'].reshape((*belief['values_std'].shape[:-2], -1)),
        belief['reservation_mean'],
        belief['reservation_std'],
    ], -1)

  def _unflatten_belief(self, flat):
    pos = 0
    weights_mean = flat[..., pos:pos + self.max_issues]
    pos += self.max_issues
    weights_std = flat[..., pos:pos + self.max_issues]
    pos += self.max_issues
    size = self.max_issues * self.max_values
    values_mean = flat[..., pos:pos + size].reshape(
        (*flat.shape[:-1], self.max_issues, self.max_values))
    pos += size
    values_std = flat[..., pos:pos + size].reshape(
        (*flat.shape[:-1], self.max_issues, self.max_values))
    pos += size
    reservation_mean = flat[..., pos:pos + 1]
    pos += 1
    reservation_std = flat[..., pos:pos + 1]
    return {
        'weights_mean': weights_mean,
        'weights_std': weights_std,
        'values_mean': values_mean,
        'values_std': values_std,
        'reservation_mean': reservation_mean,
        'reservation_std': reservation_std,
    }

  def _prior_belief_dict(self, shape):
    return {
        'weights_mean': jnp.full((*shape, self.max_issues), 1.0 / self.max_issues, f32),
        'weights_std': jnp.full((*shape, self.max_issues), 0.25, f32),
        'values_mean': jnp.full((*shape, self.max_issues, self.max_values), 0.5, f32),
        'values_std': jnp.full((*shape, self.max_issues, self.max_values), 0.5, f32),
        'reservation_mean': jnp.full((*shape, 1), 0.3, f32),
        'reservation_std': jnp.full((*shape, 1), 0.2, f32),
    }

  def _prior_belief(self, shape):
    return self._flatten_belief(self._prior_belief_dict(shape))

  def _true_belief(self, obs):
    zeros_w = jnp.zeros_like(obs['true_opp_weights'])
    zeros_v = jnp.zeros_like(obs['true_opp_values'])
    zeros_r = jnp.zeros_like(obs['true_opp_reserved'][..., None])
    return {
        'weights_mean': f32(obs['true_opp_weights']),
        'weights_std': zeros_w,
        'values_mean': f32(obs['true_opp_values']),
        'values_std': zeros_v,
        'reservation_mean': f32(obs['true_opp_reserved'][..., None]),
        'reservation_std': zeros_r,
    }

  def _mix_belief(self, pred_belief, true_belief, alpha):
    alpha = f32(alpha)
    mixed = {}
    for key, true_value in true_belief.items():
      a = alpha
      while a.ndim < true_value.ndim:
        a = a[..., None]
      if key.endswith('_std'):
        mixed[key] = a * pred_belief[key]
      else:
        mixed[key] = (1.0 - a) * true_value + a * pred_belief[key]
    return mixed

  def _score_self_offer(self, obs, offer, valid):
    offer = jnp.clip(offer, 0, self.max_values - 1)
    value = score_offer_from_table(
        offer, obs['self_weights'], obs['self_values'], obs['issue_mask'])
    return jnp.where(valid, value, jnp.zeros_like(value))

  def _opp_offer_stats(self, obs, offer, valid, belief):
    offer = jnp.clip(offer, 0, self.max_values - 1)
    chosen_mean = jnp.take_along_axis(
        belief['values_mean'], offer[..., None], axis=-1).squeeze(-1)
    chosen_std = jnp.take_along_axis(
        belief['values_std'], offer[..., None], axis=-1).squeeze(-1)
    issue_mask = obs['issue_mask'].astype(f32)
    wmean = belief['weights_mean']
    wstd = belief['weights_std']
    mean = (wmean * chosen_mean * issue_mask).sum(-1)
    var_terms = (
        jnp.square(wstd) * jnp.square(chosen_std) +
        jnp.square(wstd) * jnp.square(chosen_mean) +
        jnp.square(chosen_std) * jnp.square(wmean))
    std = jnp.sqrt(jnp.maximum((var_terms * issue_mask).sum(-1), 1e-6))
    margin = mean - belief['reservation_mean'].squeeze(-1)
    margin_std = jnp.sqrt(
        jnp.square(std) + jnp.square(belief['reservation_std'].squeeze(-1)))
    zero = jnp.zeros_like(mean)
    return (
        jnp.where(valid, mean, zero),
        jnp.where(valid, std, zero),
        jnp.where(valid, margin, zero),
        jnp.where(valid, margin_std, zero),
    )

  def _proposal_utilities(self, obs, weights, values):
    proposals = jnp.asarray(self.proposals, i32)
    return self._utilities_for_proposals(obs, weights, values, proposals)

  def _utilities_for_proposals(self, obs, weights, values, proposals):
    proposals = jnp.asarray(proposals, i32)
    if proposals.ndim == 2:
      proposals = jnp.broadcast_to(
          proposals, (*obs['issue_mask'].shape[:-1], *proposals.shape))
    offers = jnp.broadcast_to(
        proposals, (*obs['issue_mask'].shape[:-1], *proposals.shape[-2:]))
    if proposals.shape[:-2] == obs['issue_mask'].shape[:-1]:
      offers = proposals
    chosen = jnp.take_along_axis(
        values[..., None, :, :], offers[..., :, None], axis=-1).squeeze(-1)
    mask = obs['issue_mask'].astype(f32)
    return (weights[..., None, :] * chosen * mask[..., None, :]).sum(-1)

  def _proposal_uncertainty_for_proposals(self, obs, belief, proposals):
    proposals = jnp.asarray(proposals, i32)
    if proposals.ndim == 2:
      proposals = jnp.broadcast_to(
          proposals, (*obs['issue_mask'].shape[:-1], *proposals.shape))
    offers = proposals
    vmean = jnp.take_along_axis(
        belief['values_mean'][..., None, :, :],
        offers[..., :, None], axis=-1).squeeze(-1)
    vstd = jnp.take_along_axis(
        belief['values_std'][..., None, :, :],
        offers[..., :, None], axis=-1).squeeze(-1)
    mask = obs['issue_mask'].astype(f32)[..., None, :]
    wmean = belief['weights_mean'][..., None, :]
    wstd = belief['weights_std'][..., None, :]
    var_terms = (
        jnp.square(wstd) * jnp.square(vstd) +
        jnp.square(wstd) * jnp.square(vmean) +
        jnp.square(vstd) * jnp.square(wmean))
    return jnp.sqrt(jnp.maximum((var_terms * mask).sum(-1), f32(1e-6)))

  def _counts_frequency(self, obs, key):
    value_mask = obs['value_mask'].astype(f32)
    counts = jnp.maximum(f32(obs[key]), 0.0) * value_mask
    denom = jnp.maximum(counts.sum(-1, keepdims=True), f32(1.0))
    return (counts / denom).astype(f32)

  def _private_score_features(self, obs, belief, proposals):
    proposals = jnp.asarray(proposals, i32)
    if proposals.ndim == 2:
      proposals = jnp.broadcast_to(
          proposals, (*obs['issue_mask'].shape[:-1], *proposals.shape))
    prop_norm = f32(proposals) / f32(self.max_values - 1)
    issue_mask = jnp.broadcast_to(
        obs['issue_mask'].astype(f32)[..., None, :], proposals.shape)
    self_u = self._utilities_for_proposals(
        obs, obs['self_weights'], obs['self_values'], proposals)
    freq_u = self._utilities_for_proposals(
        obs, belief.freq['weights_mean'], belief.freq['values_mean'], proposals)
    pred = belief.pred()
    theta_u = self._utilities_for_proposals(
        obs, pred['weights_mean'], pred['values_mean'], proposals)
    theta_std = self._proposal_uncertainty_for_proposals(obs, pred, proposals)
    relative_time = jnp.broadcast_to(
        obs['relative_time'][..., None], theta_u.shape)
    return jnp.concatenate([
        prop_norm,
        issue_mask,
        self_u[..., None],
        freq_u[..., None],
        theta_u[..., None],
        theta_std[..., None],
        relative_time[..., None],
    ], -1), theta_u

  def _private_opp_utilities(self, feat_tensor, obs, belief, proposals=None):
    if proposals is None:
      proposals = jnp.asarray(self.proposals, i32)
    features, _ = self._private_score_features(obs, belief, proposals)
    base = 2 * self.max_issues
    theta_u = features[..., base + 2]
    return jnp.clip(theta_u, 0.0, 1.0)

  def _masked_corr_accuracy(self, true, pred, mask):
    mask = mask.astype(f32)
    count = jnp.maximum(mask.sum(-1), f32(1.0))
    true_mean = (true * mask).sum(-1) / count
    pred_mean = (pred * mask).sum(-1) / count
    true_c = (true - true_mean[..., None]) * mask
    pred_c = (pred - pred_mean[..., None]) * mask
    cov = (true_c * pred_c).sum(-1)
    true_var = jnp.square(true_c).sum(-1)
    pred_var = jnp.square(pred_c).sum(-1)
    corr = cov / jnp.sqrt(jnp.maximum(true_var * pred_var, f32(1e-12)))
    corr = jnp.where(jnp.isfinite(corr), corr, f32(0.0))
    return jnp.clip((corr + 1.0) / 2.0, 0.0, 1.0)

  def _masked_fast_kendall_accuracy(self, true, pred, mask, active=None):
    true = f32(true)
    pred = f32(pred)
    mask = mask.astype(bool)
    n = true.shape[-1]
    flat_true = true.reshape((-1, n))
    flat_pred = pred.reshape((-1, n))
    flat_mask = mask.reshape((-1, n))
    if active is None:
      acc = jax.vmap(self._fast_kendall_1d)(flat_true, flat_pred, flat_mask)
    else:
      flat_active = active.astype(bool).reshape((-1,))

      def compute_one(inputs):
        t, p, m, a = inputs
        return jax.lax.cond(
            a,
            lambda _: self._fast_kendall_1d(t, p, m),
            lambda _: f32(0.0),
            operand=None)

      acc = jax.lax.map(compute_one, (flat_true, flat_pred, flat_mask, flat_active))
    return acc.reshape(true.shape[:-1])

  def _fast_kendall_1d(self, true, pred, mask):
    n = true.shape[0]
    idx = jnp.arange(n, dtype=i32)
    valid_count = mask.astype(i32).sum()
    true_key = jnp.where(mask, true, jnp.inf)
    pred_key = jnp.where(mask, pred, jnp.inf)
    true_order = jnp.argsort(true_key, stable=True)
    pred_order = jnp.argsort(pred_key, stable=True)
    pred_rank = jnp.zeros((n,), i32).at[pred_order].set(idx)
    seq = pred_rank[true_order]
    valid_seq = mask[true_order]

    def bit_query(bit, pos):
      def cond(state):
        _, i, _ = state
        return i > 0
      def body(state):
        bit, i, total = state
        total = total + bit[i]
        i = i - (i & -i)
        return bit, i, total
      _, _, total = jax.lax.while_loop(
          cond, body, (bit, pos.astype(i32), f32(0.0)))
      return total

    def bit_update(bit, pos, value):
      def cond(state):
        _, i = state
        return i <= n
      def body(state):
        bit, i = state
        bit = bit.at[i].add(value)
        i = i + (i & -i)
        return bit, i
      bit, _ = jax.lax.while_loop(
          cond, body, (bit, pos.astype(i32),))
      return bit

    def step(carry, inputs):
      bit, seen, inv = carry
      rank, valid = inputs
      valid_f = valid.astype(f32)
      pos = rank + 1
      leq = bit_query(bit, pos)
      inv = inv + valid_f * (seen - leq)
      bit = bit_update(bit, pos, valid_f)
      seen = seen + valid_f
      return (bit, seen, inv), None

    bit0 = jnp.zeros((n + 1,), f32)
    (_, _, inversions), _ = jax.lax.scan(
        step, (bit0, f32(0.0), f32(0.0)), (seq, valid_seq))
    pairs = f32(valid_count * (valid_count - 1)) / 2.0
    acc = 1.0 - inversions / jnp.maximum(pairs, f32(1.0))
    return jnp.where(valid_count >= 2, jnp.clip(acc, 0.0, 1.0), f32(0.0))

  def _frequency_belief_from_self_offers(self, obs):
    value_mask = obs['value_mask'].astype(f32)
    counts = jnp.maximum(f32(obs['self_offer_counts']), 0.0)
    counts = counts * value_mask
    totals = jnp.maximum(counts.sum(-1, keepdims=True), f32(1.0))
    values = counts / totals
    values = jnp.where(value_mask > 0, values, 0.0)
    centered = values - (
        (values * value_mask).sum(-1, keepdims=True) /
        jnp.maximum(value_mask.sum(-1, keepdims=True), f32(1.0)))
    spread = jnp.sqrt(jnp.maximum(
        (jnp.square(centered) * value_mask).sum(-1) /
        jnp.maximum(value_mask.sum(-1), f32(1.0)), f32(0.0)))
    active = obs['issue_mask'].astype(f32)
    spread = jnp.where(active > 0, spread, 0.0)
    denom = spread.sum(-1, keepdims=True)
    uniform = active / jnp.maximum(active.sum(-1, keepdims=True), 1.0)
    weights = jnp.where(denom > 1e-9, spread / denom, uniform)
    return weights, values

  def _theta_anac_reward(self, feat_tensor, obs):
    if self.reward_target not in ('anac_theta_score', 'anac_kendall_score'):
      return obs['reward']
    if self.reward_target == 'anac_kendall_score':
      return obs.get('dreamer_cached_reward', obs['reward'])
    belief = jax.tree.map(sg, self.osi(feat_tensor, obs, 2).pred())
    pred_opp = self._proposal_utilities(
        obs, belief['weights_mean'], belief['values_mean'])
    true_opp = self._proposal_utilities(
        obs, f32(obs['true_opp_weights']), f32(obs['true_opp_values']))

    opp_w, opp_v = self._frequency_belief_from_self_offers(obs)
    opp_est_self = self._proposal_utilities(obs, opp_w, opp_v)
    true_self = self._proposal_utilities(
        obs, f32(obs['self_weights']), f32(obs['self_values']))
    terminal = obs['is_terminal'].astype(f32)
    my_acc = self._masked_corr_accuracy(
        true_opp, pred_opp, obs['proposal_mask'])
    opp_acc = self._masked_corr_accuracy(
        true_self, opp_est_self, obs['proposal_mask'])

    concealing = my_acc / jnp.maximum(my_acc + opp_acc, f32(1e-6))
    agreement = terminal * (
        (jnp.abs(obs['reward']) > 1e-8).astype(f32))
    anac_score = obs['reward'] + terminal * concealing
    delay_bonus = 0.5 * jnp.power(
        jnp.clip(f32(obs['relative_time']), 0.0, 1.0), 1.75)
    return anac_score * (1.0 + agreement * delay_bonus)

  def _theta_my_accuracy(self, obs, belief):
    pred_opp = self._proposal_utilities(
        obs, belief['weights_mean'], belief['values_mean'])
    true_opp = self._proposal_utilities(
        obs, f32(obs['true_opp_weights']), f32(obs['true_opp_values']))
    return self._masked_corr_accuracy(true_opp, pred_opp, obs['proposal_mask'])

  def _cached_dreamer_reward(self, obs, belief):
    pred = jax.tree.map(sg, belief.pred())
    pred_opp = self._proposal_utilities(
        obs, pred['weights_mean'], pred['values_mean'])
    true_opp = self._proposal_utilities(
        obs, f32(obs['true_opp_weights']), f32(obs['true_opp_values']))
    opp_w, opp_v = self._frequency_belief_from_self_offers(obs)
    opp_est_self = self._proposal_utilities(obs, opp_w, opp_v)
    true_self = self._proposal_utilities(
        obs, f32(obs['self_weights']), f32(obs['self_values']))
    terminal = obs['is_terminal'].astype(f32)
    active = terminal > 0.5
    my_acc = self._masked_fast_kendall_accuracy(
        true_opp, pred_opp, obs['proposal_mask'], active)
    opp_acc = self._masked_fast_kendall_accuracy(
        true_self, opp_est_self, obs['proposal_mask'], active)
    concealing = my_acc / jnp.maximum(my_acc + opp_acc, f32(1e-6))
    reward = obs['reward'] + terminal * concealing
    return reward.astype(f32), my_acc.astype(f32), opp_acc.astype(f32), concealing.astype(f32)

  def _offer_semantic_vec(self, obs, offer, valid, belief):
    offer = jnp.clip(offer, 0, self.max_values - 1)
    issue_mask = obs['issue_mask'].astype(f32)
    active = issue_mask * valid.astype(f32)[..., None]
    self_value = jnp.take_along_axis(
        obs['self_values'], offer[..., :, None], -1).squeeze(-1)
    opp_value_mean = jnp.take_along_axis(
        belief['values_mean'], offer[..., :, None], -1).squeeze(-1)
    opp_value_std = jnp.take_along_axis(
        belief['values_std'], offer[..., :, None], -1).squeeze(-1)
    return jnp.concatenate([
        obs['self_weights'] * active,
        self_value * active,
        belief['weights_mean'] * active,
        belief['weights_std'] * active,
        opp_value_mean * active,
        opp_value_std * active,
    ], -1)

  def _encoder_obs_vec(self, obs, prev_belief, prevact):
    belief = self._unflatten_belief(sg(prev_belief))
    last_opp_valid = obs['last_opp_valid'].astype(bool)
    last_self_valid = obs['last_self_valid'].astype(bool)
    last_opp = jnp.clip(obs['last_opp_offer'], 0, self.max_values - 1)
    last_self = jnp.clip(obs['last_self_offer'], 0, self.max_values - 1)
    last_opp_norm = f32(last_opp) / f32(self.max_values - 1)
    last_self_norm = f32(last_self) / f32(self.max_values - 1)
    last_opp_norm = jnp.where(last_opp_valid[..., None], last_opp_norm, 0.0)
    last_self_norm = jnp.where(last_self_valid[..., None], last_self_norm, 0.0)
    self_u_last_opp = self._score_self_offer(obs, last_opp, last_opp_valid)
    self_u_last_self = self._score_self_offer(obs, last_self, last_self_valid)
    self_margin_last_opp = self_u_last_opp - obs['self_reserved']
    self_margin_last_self = self_u_last_self - obs['self_reserved']
    self_margin_last_opp = jnp.where(
        last_opp_valid, self_margin_last_opp, jnp.zeros_like(self_margin_last_opp))
    self_margin_last_self = jnp.where(
        last_self_valid, self_margin_last_self, jnp.zeros_like(self_margin_last_self))
    opp_last = self._opp_offer_stats(obs, last_opp, last_opp_valid, belief)
    opp_self = self._opp_offer_stats(obs, last_self, last_self_valid, belief)
    last_opp_semantic = self._offer_semantic_vec(
        obs, last_opp, last_opp_valid, belief)
    last_self_semantic = self._offer_semantic_vec(
        obs, last_self, last_self_valid, belief)
    issue_mask = obs['issue_mask'].astype(f32)
    value_mask = obs['value_mask'].astype(f32)
    return jnp.concatenate([
        obs['relative_time'][..., None],
        sg(self._target_value_context(prevact)),
        last_opp_norm,
        last_self_norm,
        last_opp_valid.astype(f32)[..., None],
        last_self_valid.astype(f32)[..., None],
        self_u_last_opp[..., None],
        self_margin_last_opp[..., None],
        self_u_last_self[..., None],
        self_margin_last_self[..., None],
        *[x[..., None] for x in opp_last],
        *[x[..., None] for x in opp_self],
        issue_mask,
        value_mask.reshape((*value_mask.shape[:-2], -1)),
        last_opp_semantic,
        last_self_semantic,
    ], -1)

  def _encoder_obs(self, obs, prev_belief, prevact):
    obs = obs.copy()
    obs['obs'] = self._encoder_obs_vec(obs, prev_belief, prevact)
    return obs

  def _proposal_aux(self, obs, belief, bdims):
    proposals = jnp.asarray(self.proposals, i32)
    proposal_mask = proposal_mask_from_obs(proposals, obs)
    self_u = score_proposals_from_table(
        proposals, obs['self_weights'], obs['self_values'], obs['issue_mask'])
    opp_u, opp_std, opp_margin, opp_margin_std = proposal_utility_stats(
        proposals,
        belief['weights_mean'], belief['weights_std'],
        belief['values_mean'], belief['values_std'],
        belief['reservation_mean'], belief['reservation_std'],
        obs['issue_mask'])
    return {
        'proposal_mask': proposal_mask,
        'self_u': self_u,
        'opp_u_mean': opp_u,
        'opp_u_std': opp_std,
        'opp_margin_mean': opp_margin,
        'opp_margin_std': opp_margin_std,
    }

  def _head_input_from_belief(
      self, feat_tensor, obs, belief, bdims, return_aux=False):
    parts, metrics = self._head_parts_from_belief(
        feat_tensor, obs, belief, bdims)
    if return_aux:
      return parts['head'], metrics
    return parts['head']

  def _head_parts_from_belief(self, feat_tensor, obs, belief, bdims):
    issue_mask = obs['issue_mask'].astype(f32)
    value_mask = obs['value_mask'].astype(f32)
    self_offer_freq = self._counts_frequency(obs, 'self_offer_counts')
    global_extra = jnp.concatenate([
        obs['self_weights'],
        obs['self_values'].reshape((*obs['self_values'].shape[:-2], -1)),
        belief['reservation_mean'],
        belief['reservation_std'],
        belief['weights_mean'],
        belief['weights_std'],
        belief['values_mean'].reshape((*belief['values_mean'].shape[:-2], -1)),
        belief['values_std'].reshape((*belief['values_std'].shape[:-2], -1)),
        issue_mask,
        value_mask.reshape((*value_mask.shape[:-2], -1)),
        self_offer_freq.reshape((*self_offer_freq.shape[:-2], -1)),
    ], -1)
    proposal_aux = self._proposal_aux(obs, belief, bdims)
    metrics = {'proposal/valid_rate': proposal_aux['proposal_mask'].mean()}
    type_vec = belief.get('type_probs')
    if type_vec is None:
      type_vec = jnp.zeros((*feat_tensor.shape[:bdims], self.config.varibad.num_tasks), f32)
    type_vec = sg(type_vec)
    fused = self.fusion(
        feat_tensor, global_extra, bdims, type_vec)
    head_input = jnp.concatenate([
        fused,
        obs['relative_time'][..., None].astype(f32),
        obs['self_reserved'][..., None].astype(f32),
    ], -1)
    metrics.update({
        'osi_type/entropy': (
            -(type_vec * jnp.log(jnp.maximum(type_vec, f32(1e-8)))).sum(-1).mean()),
        'osi_type/max_prob': type_vec.max(-1).mean(),
    })
    parts = {
        'head': head_input,
        'obs': obs,
        'proposal_aux': proposal_aux,
    }
    return parts, metrics

  def _head_input(self, feat_tensor, obs, bdims, alpha, return_aux=False):
    belief = self.osi(feat_tensor, obs, bdims)
    pred_belief = jax.tree.map(sg, belief.pred())
    mixed = self._mix_belief(
        pred_belief, self._true_belief(obs), alpha)
    mixed['type_probs'] = pred_belief['type_probs']
    return self._head_input_from_belief(
        feat_tensor, obs, mixed, bdims, return_aux)

  def _head_parts(self, feat_tensor, obs, bdims, alpha, return_aux=False):
    belief = self.osi(feat_tensor, obs, bdims)
    pred_belief = jax.tree.map(sg, belief.pred())
    mixed = self._mix_belief(
        pred_belief, self._true_belief(obs), alpha)
    mixed['type_probs'] = pred_belief['type_probs']
    parts, metrics = self._head_parts_from_belief(
        feat_tensor, obs, mixed, bdims)
    if return_aux:
      return parts, metrics
    return parts

  def _imag_head_input(self, feat_tensor, obs, bdims, alpha, return_aux=False):
    belief = self.osi(feat_tensor, obs, bdims)
    pred_belief = jax.tree.map(sg, belief.pred())
    mixed = self._mix_belief(
        pred_belief, self._true_belief(obs), alpha)
    mixed['type_probs'] = pred_belief['type_probs']
    return self._head_input_from_belief(
        feat_tensor, obs, mixed, bdims, return_aux)

  def _imag_head_parts(self, feat_tensor, obs, bdims, alpha, return_aux=False):
    belief = self.osi(feat_tensor, obs, bdims)
    pred_belief = jax.tree.map(sg, belief.pred())
    mixed = self._mix_belief(
        pred_belief, self._true_belief(obs), alpha)
    mixed['type_probs'] = pred_belief['type_probs']
    parts, metrics = self._head_parts_from_belief(
        feat_tensor, obs, mixed, bdims)
    if return_aux:
      return parts, metrics
    return parts

  def _make_proposals(self, obs_space, act_space):
    n_actions = int(np.asarray(act_space['action'].classes).flatten()[0])
    n_proposals = n_actions - 1
    obs_dim = obs_space['obs'].shape[0]
    n_issues = int((obs_dim - 5) // 4)
    issue_size = round(n_proposals ** (1.0 / n_issues))
    issue_sizes = [issue_size] * n_issues
    assert int(np.prod(issue_sizes)) == n_proposals, (issue_sizes, n_proposals)
    outcomes = np.array(
        np.meshgrid(*[np.arange(s) for s in issue_sizes], indexing='ij')
    ).reshape(n_issues, -1).T
    return outcomes.astype(np.float32) / (
        np.asarray(issue_sizes, np.float32) - 1.0)

  def _dyn_action(self, act, obs, bdims):
    return {
        'action_emb': self.action_enc(
            act['action'], obs, bdims, self._target_raw_context(act))}

  def _target_context(self, act):
    base = act['action']
    zero = jnp.zeros(base.shape, f32)
    keys = (
        'target_z_e',
        'target_z_delta_down',
        'target_z_delta_up',
        'target_z_umax',
        'target_rho',
        'target_delta_down',
        'target_delta_up',
        'target_u_max',
    )
    vals = [f32(act[key] if key in act else zero)[..., None] for key in keys]
    return jnp.concatenate(vals, -1)

  def _target_raw_context(self, act):
    return self._target_context(act)[..., :4]

  def _target_value_context(self, act):
    return self._target_context(act)[..., 4:]

  def _augment_policy_parts(self, parts, prevact):
    parts = dict(parts)
    parts['head'] = jnp.concatenate(
        [parts['head'], sg(self._target_context(prevact))], -1)
    return parts

  def _attach_target_action_info(self, act, policy, deterministic=False):
    act = dict(act)
    band = policy['action']
    if deterministic:
      act['target_z_e'] = band.mu_e.squeeze(-1)
      act['target_z_delta_down'] = band.mu_delta_down.squeeze(-1)
      act['target_z_delta_up'] = band.mu_delta.squeeze(-1)
      act['target_z_umax'] = band.mu_umax.squeeze(-1)
      act['target_rho'] = band.det_rho.squeeze(-1)
      act['target_delta_down'] = band.det_delta_down.squeeze(-1)
      act['target_delta_up'] = band.det_delta.squeeze(-1)
      act['target_u_max'] = band.det_u_max.squeeze(-1)
    else:
      act['target_z_e'] = band.z_e.squeeze(-1)
      act['target_z_delta_down'] = band.z_delta_down.squeeze(-1)
      act['target_z_delta_up'] = band.z_delta.squeeze(-1)
      act['target_z_umax'] = band.z_umax.squeeze(-1)
      act['target_rho'] = band.rho.squeeze(-1)
      act['target_delta_down'] = band.delta_down.squeeze(-1)
      act['target_delta_up'] = band.delta.squeeze(-1)
      act['target_u_max'] = band.u_max.squeeze(-1)
    return act

  def _attach_dreamer_table_action_info(self, act, belief):
    act = dict(act)
    act['dreamer_opp_weights'] = belief['weights_mean']
    act['dreamer_opp_values'] = belief['values_mean']
    return act

  def _offer_value_input(self, head, act):
    return jnp.concatenate([head, sg(self._target_value_context(act))], -1)

  def _imagine_env_actions(self, starts, obs, prevact, alpha, length, training):
    def step(carry, _):
      state, prev = carry
      feat_tensor = self.feat2tensor(state)
      belief = self.osi(feat_tensor, obs, 1)
      pred_belief = jax.tree.map(sg, belief.pred())
      mixed = self._mix_belief(
          pred_belief, self._true_belief(obs), alpha)
      mixed['type_probs'] = pred_belief['type_probs']
      parts, _ = self._head_parts_from_belief(feat_tensor, obs, mixed, 1)
      parts = self._augment_policy_parts(parts, prev)
      policy = self.pol(parts['head'], parts['obs'], parts['proposal_aux'], 1)
      env_action = self._attach_target_action_info(
          sample(policy), policy, deterministic=False)
      env_action = self._attach_dreamer_table_action_info(env_action, mixed)
      dyn_action = self._dyn_action(env_action, obs, 1)
      next_state, (next_feat, _) = self.dyn.imagine(
          state, dyn_action, 1, training, single=True)
      return (next_state, env_action), (next_feat, env_action)

    unroll = length if self.dyn.unroll else 1
    (carry, _), (feat, action) = nj.scan(
        step, (nn.cast(starts), prevact), (), length, unroll=unroll, axis=1)
    return carry, feat, action

  def _posterior_with_encoder_input(
      self, carry, obs, prevact, reset, alpha, training):
    enc_carry, dyn_carry, dec_carry = carry
    B, T = reset.shape
    prior = self._prior_belief((B, T))
    enc_obs = self._encoder_obs(obs, prior, prevact)
    enc_carry0, _, tokens0 = self.enc(enc_carry, enc_obs, reset, training)
    dyn_prevact = self._dyn_action(prevact, obs, 2)
    dyn_carry0, _, _, repfeat0, _ = self.dyn.loss(
        dyn_carry, tokens0, dyn_prevact, reset, training)
    belief0 = self.osi(self.feat2tensor(repfeat0), obs, 2)
    mixed = self._mix_belief(belief0.pred(), self._true_belief(obs), alpha)
    belief_seq = sg(self._flatten_belief(mixed))
    prior0 = self._prior_belief((B,))[:, None]
    prev_belief = jnp.concatenate([prior0, belief_seq[:, :-1]], 1)
    prev_belief = nn.where(reset[..., None], prior, prev_belief)
    enc_obs = self._encoder_obs(obs, prev_belief, prevact)
    obs['obs'] = enc_obs['obs']
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, enc_obs, reset, training)
    dyn_prevact = self._dyn_action(prevact, obs, 2)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, dyn_prevact, reset, training)
    return (
        (enc_carry, dyn_carry, dec_carry),
        (enc_entries, dyn_entries),
        tokens,
        los,
        repfeat,
        mets,
        belief_seq[:, -1])

  def _policy_input(self, feat, theta):
    return jnp.concatenate([self.feat2tensor(feat), theta], -1)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact, prev_belief, policy_step) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    prior = self._prior_belief(prev_belief.shape[:-1])
    prev_belief = nn.where(reset[..., None], prior, prev_belief)
    enc_obs = self._encoder_obs(obs, prev_belief, prevact)
    enc_carry, enc_entry, tokens = self.enc(enc_carry, enc_obs, reset, **kw)
    dyn_prevact = self._dyn_action(prevact, obs, 1)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, dyn_prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    feat_tensor = self.feat2tensor(feat)
    belief = self.osi(feat_tensor, obs, 1)
    pred_belief = belief.pred()
    if self.config.osi.detach_actor:
      pred_belief = jax.tree.map(sg, pred_belief)
    alpha = self._policy_alpha(policy_step, mode, obs)
    mixed_belief = self._mix_belief(
        jax.tree.map(sg, pred_belief), self._true_belief(obs), alpha)
    mixed_belief['type_probs'] = pred_belief['type_probs']
    parts, _ = self._head_parts_from_belief(
        feat_tensor, obs, mixed_belief, 1)
    parts = self._augment_policy_parts(parts, prevact)
    policy = self.pol(
        parts['head'], parts['obs'], parts['proposal_aux'], bdims=1)
    act = pred(policy) if mode.startswith('eval') else sample(policy)
    act = self._attach_target_action_info(
        act, policy, deterministic=mode.startswith('eval'))
    act = self._attach_dreamer_table_action_info(act, mixed_belief)
    next_belief = self._flatten_belief(pred_belief)
    out = {}
    if mode in ('train', 'eval', 'eval_osi'):
      def compute_private_utils():
        return self._private_opp_utilities(feat_tensor, obs, belief)

      def zero_private_utils():
        shape = (*obs['is_terminal'].shape, self.proposals.shape[0])
        return jnp.zeros(shape, f32)

      out['private_opp_utils'] = jax.lax.cond(
          jnp.any(obs['is_terminal']),
          compute_private_utils,
          zero_private_utils)
    if mode == 'eval_osi':
      out['opp_belief'] = next_belief
    if mode.startswith('eval'):
      theta_my_acc = self._theta_my_accuracy(obs, pred_belief)
      agreement = obs['is_terminal'].astype(f32) * (
          (jnp.abs(obs['reward']) > 1e-8).astype(f32))
      out['log/theta_my_accuracy_terminal'] = theta_my_acc * agreement
      out['theta_weights_mean'] = pred_belief['weights_mean']
      out['theta_values_mean'] = pred_belief['values_mean']
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (
        enc_carry, dyn_carry, dec_carry, act, next_belief,
        policy_step + f32(1.0))
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, act, stepid, prev_belief, policy_step = self._apply_replay_context(
        carry, data)
    alpha = self._osi_alpha(data)
    metrics, (carry, entries, loss_outs, mets) = self.opt(
        self.loss, carry, obs, prevact, act, prev_belief, alpha,
        training=True, has_aux=True)
    metrics.update(mets)
    self.slowval_target.update()
    self.slowval_offer.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (
        *carry,
        {k: data[k][:, -1] for k in self.act_space},
        loss_outs['last_belief'],
        policy_step)
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, act, prev_belief, alpha, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}
    metrics['osi_alpha'] = alpha

    # World model
    carry, entries, tokens, los, repfeat, mets, last_belief = (
        self._posterior_with_encoder_input(
            carry, obs, prevact, reset, alpha, training))
    enc_carry, dyn_carry, dec_carry = carry
    enc_entries, dyn_entries = entries
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    feat_tensor = self.feat2tensor(repfeat)
    belief = self.osi(sg(feat_tensor), obs, 2)
    losses['osi'] = jnp.zeros((B, T), f32) + belief.loss(obs)
    metrics.update(belief.metrics(obs))
    reward_target = self._theta_anac_reward(feat_tensor, obs)
    metrics['reward/target_mean'] = reward_target.mean()
    metrics['reward/env_mean'] = obs['reward'].mean()
    feat_rew = sg(feat_tensor, skip=self.config.reward_grad)
    rewcon_inp, landscape_metrics = self._head_input(
        feat_rew, obs, 2, alpha, return_aux=True)
    metrics.update(landscape_metrics)
    losses['rew'] = self.rew(rewcon_inp, 2).loss(reward_target)
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    feat_con = sg(feat_tensor, skip=self.config.reward_grad)
    con_inp = self._head_input(feat_con, obs, 2, alpha)
    losses['con'] = self.con(con_inp, 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    obs_start = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, *x.shape[2:])), obs)
    prevact_start = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, *x.shape[2:])), prevact)

    def policyfn(feat, prev):
      feat_tensor = self.feat2tensor(feat)
      belief = self.osi(feat_tensor, obs_start, 1)
      pred_belief = jax.tree.map(sg, belief.pred())
      mixed = self._mix_belief(
          pred_belief, self._true_belief(obs_start), alpha)
      mixed['type_probs'] = pred_belief['type_probs']
      parts, _ = self._head_parts_from_belief(
          feat_tensor, obs_start, mixed, 1)
      parts = self._augment_policy_parts(parts, prev)
      policy = self.pol(parts['head'], parts['obs'], parts['proposal_aux'], 1)
      act = self._attach_target_action_info(
          sample(policy), policy, deterministic=False)
      return self._attach_dreamer_table_action_info(act, mixed)

    _, imgfeat, imgprevact = self._imagine_env_actions(
        starts, obs_start, prevact_start, alpha, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    last_prevact = jax.tree.map(lambda x: x[:, -1], imgprevact)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat), last_prevact)
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    obs_img = jax.tree.map(
        lambda x: jnp.repeat(x[:, None], H + 1, axis=1), obs_start)
    head_img_parts, img_landscape_metrics = self._imag_head_parts(
        inp, obs_img, 2, alpha, return_aux=True)
    img_policy_prevact = jax.tree.map(
        lambda start, acts: jnp.concatenate([start[:, None], acts[:, :-1]], 1),
        prevact_start, imgact)
    policy_img_parts = self._augment_policy_parts(
        head_img_parts, img_policy_prevact)
    head_img_inp = head_img_parts['head']
    offer_img_inp = self._offer_value_input(head_img_inp, imgact)
    metrics.update(prefix(img_landscape_metrics, 'imag'))
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(head_img_inp, 2).pred(),
        self.con(head_img_inp, 2).prob(1),
        self.pol(
            policy_img_parts['head'], policy_img_parts['obs'],
            policy_img_parts['proposal_aux'], 2),
        self.val_target(head_img_inp, 2),
        self.slowval_target(head_img_inp, 2),
        self.val_offer(offer_img_inp, 2),
        self.slowval_offer(offer_img_inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term = [obs[k] for k in ('is_last', 'is_terminal')]
      rew = reward_target
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      obs_rep = jax.tree.map(lambda x: x[:, -K:], obs)
      rep_inp = self._head_input(inp, obs_rep, 2, alpha)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(rep_inp, 2),
          self.slowval(rep_inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()).issubset(set(self.scales.keys())), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    outs['last_belief'] = last_belief
    return loss, (carry, entries, outs, metrics)

  def _varibad_action_features(self, act):
    action = act['action']
    proposals = jnp.asarray(self.proposals, i32)
    n_props = proposals.shape[0]
    is_accept = action >= n_props
    proposal_id = jnp.clip(action, 0, n_props - 1)
    offer = jnp.take(proposals, proposal_id, axis=0)
    offer_norm = f32(offer) / f32(self.max_values - 1)
    offer_norm = jnp.where(is_accept[..., None], 0.0, offer_norm)
    return jnp.concatenate([
        is_accept.astype(f32)[..., None],
        (~is_accept).astype(f32)[..., None],
        offer_norm,
    ], -1)

  def _varibad_decoder_input(self, z, feat, act_onehot):
    return jnp.concatenate([z, feat, act_onehot], -1)

  def _varibad_loss(self, feat_tensor, act, obs, reward):
    B, T = obs['is_first'].shape
    if T < 2:
      zero = jnp.zeros((), f32)
      zero_bt = jnp.zeros((B, T), f32)
      return {
          'varibad_reward': zero_bt,
          'varibad_offer': zero_bt,
          'varibad_task': zero_bt,
          'varibad_kl': zero_bt,
      }, {}

    task = self.task_belief(sg(feat_tensor), 2)
    z = task.z
    feat_k = sg(feat_tensor[:, :-1])
    act_k = self._varibad_action_features(act)[:, :-1]
    reward_target = reward[:, 1:]
    offer_target = jnp.clip(
        obs['last_opp_offer'][:, 1:], 0, self.max_values - 1)
    offer_valid = obs['last_opp_valid'][:, 1:].astype(f32)

    z_i = z[:, :, None, :]
    feat_k = feat_k[:, None]
    act_k = act_k[:, None]
    z_i = z_i + jnp.zeros((*z_i.shape[:2], T - 1, z_i.shape[-1]), f32)
    feat_k = feat_k + jnp.zeros((z.shape[0], T, T - 1, feat_k.shape[-1]), f32)
    act_k = act_k + jnp.zeros((z.shape[0], T, T - 1, act_k.shape[-1]), f32)
    dec_inp = self._varibad_decoder_input(z_i, feat_k, act_k)

    reward_pred = self.varibad_reward(dec_inp, 3).squeeze(-1)
    reward_loss = jnp.square(reward_pred - reward_target[:, None]).mean()

    offer_logits = self.varibad_offer(dec_inp, 3).reshape(
        (B, T, T - 1, self.max_issues, self.max_values))
    value_mask = obs['value_mask'][:, 1:].astype(bool)
    value_mask = value_mask[:, None]
    offer_logits = jnp.where(
        value_mask, offer_logits, jnp.full_like(offer_logits, -1e9))
    offer_logprob = jax.nn.log_softmax(offer_logits, -1)
    offer_onehot = jax.nn.one_hot(
        offer_target[:, None], self.max_values, dtype=f32)
    offer_ce = -(offer_onehot * offer_logprob).sum(-1)
    issue_mask = obs['issue_mask'][:, 1:].astype(f32)
    offer_mask = offer_valid[:, None, :, None] * issue_mask[:, None]
    offer_loss = (
        (offer_ce * offer_mask).sum() /
        jnp.maximum(f32(1.0), offer_mask.sum()))

    task_logits = self.varibad_task(z, 2)
    task_target = obs['opponent_type_id']
    task_logprob = jax.nn.log_softmax(task_logits, -1)
    task_onehot = jax.nn.one_hot(
        task_target, self.config.varibad.num_tasks, dtype=f32)
    task_loss = -(task_onehot * task_logprob).sum(-1).mean()

    kl_loss = task.kl_standard_normal().mean()
    losses = {
        'varibad_reward': jnp.zeros((B, T), f32) + reward_loss,
        'varibad_offer': jnp.zeros((B, T), f32) + offer_loss,
        'varibad_task': jnp.zeros((B, T), f32) + task_loss,
        'varibad_kl': jnp.zeros((B, T), f32) + kl_loss,
    }
    metrics = {
        'varibad/reward_loss': reward_loss,
        'varibad/offer_loss': offer_loss,
        'varibad/task_loss': task_loss,
        'varibad/kl': kl_loss,
        'varibad/offer_valid_rate': offer_valid.mean(),
        'varibad/task_acc': (
            jnp.argmax(task_logits, -1) == task_target).mean(),
    }
    return losses, metrics

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, act, _, prev_belief, policy_step = self._apply_replay_context(
        carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    alpha = self._osi_alpha(data)
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, act, prev_belief, alpha, training=False)
    metrics.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, act, prev_belief, alpha,
              training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_first_prevact = self._dyn_action(firsthalf(prevact), firsthalf(obs), 2)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), dyn_first_prevact,
        firsthalf(obs['is_first']), training=False)
    dyn_second_prevact = self._dyn_action(
        secondhalf(prevact), secondhalf(obs), 2)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, dyn_second_prevact, length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (
        *new_carry,
        {k: data[k][:, -1] for k in self.act_space},
        outs['last_belief'],
        policy_step)
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact, prev_belief, policy_step) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    data_obs_keys = [k for k in self.obs_space if k != 'obs']
    obs = {k: data[k] for k in data_obs_keys}
    for key in (
        'dreamer_cached_reward', 'dreamer_my_acc',
        'dreamer_opp_acc', 'dreamer_concealing'):
      if key in data:
        obs[key] = data[key]
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    act = {k: data[k] for k in self.act_space}
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    belief_prev = self._prior_belief(obs['is_first'].shape[:1])
    if not self.config.replay_context:
      return carry, obs, prevact, act, stepid, belief_prev, policy_step

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in data_obs_keys}
    for key in (
        'dreamer_cached_reward', 'dreamer_my_acc',
        'dreamer_opp_acc', 'dreamer_concealing'):
      if key in data:
        rep_obs[key] = rhs(data[key])
    rep_act = {k: rhs(data[k]) for k in self.act_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)
    rep_belief_prev = self._prior_belief(rep_obs['is_first'].shape[:1])

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, act, stepid, belief_prev = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(act), rhs(stepid), belief_prev),
        (rep_carry, rep_obs, rep_prevact, rep_act, rep_stepid,
         rep_belief_prev))
    return carry, obs, prevact, act, stepid, belief_prev, policy_step

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value_target, slowvalue_target, value_offer, slowvalue_offer,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val_target = value_target.pred() * vscale + voffset
  slowval_target = slowvalue_target.pred() * vscale + voffset
  tarval_target = slowval_target if slowtar else val_target
  val_offer = value_offer.pred() * vscale + voffset
  slowval_offer = slowvalue_offer.pred() * vscale + voffset
  tarval_offer = slowval_offer if slowtar else val_offer
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret_target = lambda_return(
      last, term, rew, tarval_target, tarval_target, disc, lam)
  ret_offer = lambda_return(
      last, term, rew, tarval_offer, tarval_offer, disc, lam)

  roffset, rscale = retnorm(ret_target, update)
  adv_target = (ret_target - tarval_target[:, :-1]) / rscale
  adv_offer = (ret_offer - tarval_offer[:, :-1]) / rscale
  adv_both = jnp.concatenate([adv_target, adv_offer], 0)
  aoffset, ascale = advnorm(adv_both, update)
  adv_target_normed = (adv_target - aoffset) / ascale
  adv_offer_normed = (adv_offer - aoffset) / ascale
  band_logpi = sum([
      v.band_logp()[:, :-1] if hasattr(v, 'band_logp')
      else jnp.zeros_like(rew[:, :-1])
      for v in policy.values()])
  action_logpi = sum([
      v.action_logp(sg(act[k]))[:, :-1] if hasattr(v, 'action_logp')
      else v.logp(sg(act[k]))[:, :-1]
      for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      band_logpi * sg(adv_target_normed) +
      action_logpi * sg(adv_offer_normed) +
      actent * sum(ents.values()))
  losses['policy'] = policy_loss

  valnorm_input = jnp.concatenate([ret_target, ret_offer], 0)
  voffset, vscale = valnorm(valnorm_input, update)
  tar_target_normed = (ret_target - voffset) / vscale
  tar_offer_normed = (ret_offer - voffset) / vscale
  tar_target_padded = jnp.concatenate(
      [tar_target_normed, 0 * tar_target_normed[:, -1:]], 1)
  tar_offer_padded = jnp.concatenate(
      [tar_offer_normed, 0 * tar_offer_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value_target.loss(sg(tar_target_padded)) +
      slowreg * value_target.loss(sg(slowvalue_target.pred())) +
      value_offer.loss(sg(tar_offer_padded)) +
      slowreg * value_offer.loss(sg(slowvalue_offer.pred())))[:, :-1]

  ret_normed = (ret_target - roffset) / rscale
  metrics['adv'] = adv_target.mean()
  metrics['adv_std'] = adv_target.std()
  metrics['adv_mag'] = jnp.abs(adv_target).mean()
  metrics['adv_target'] = adv_target.mean()
  metrics['adv_offer'] = adv_offer.mean()
  metrics['adv_target_mag'] = jnp.abs(adv_target).mean()
  metrics['adv_offer_mag'] = jnp.abs(adv_offer).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['ret_target'] = ret_normed.mean()
  metrics['ret_offer'] = ((ret_offer - roffset) / rscale).mean()
  metrics['val'] = val_target.mean()
  metrics['val_target'] = val_target.mean()
  metrics['val_offer'] = val_offer.mean()
  metrics['tar'] = tar_target_normed.mean()
  metrics['tar_target'] = tar_target_normed.mean()
  metrics['tar_offer'] = tar_offer_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval_target.mean()
  metrics['slowval_target'] = slowval_target.mean()
  metrics['slowval_offer'] = slowval_offer.mean()
  metrics['logpi_band'] = band_logpi.mean()
  metrics['logpi_action'] = action_logpi.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in policy:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret_target
  outs['ret_target'] = ret_target
  outs['ret_offer'] = ret_offer
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
