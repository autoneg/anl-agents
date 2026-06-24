import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.outs as outs
import embodied.jax.nets as nn
from . import rssm


f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


def fixed_5x6_proposals():
  outcomes = np.array(
      np.meshgrid(*[np.arange(5) for _ in range(6)], indexing='ij')
  ).reshape(6, -1).T
  return outcomes.astype(np.int32)


class MaskedPMA(nj.Module):

  def __init__(self, units=128, num_seeds=4, heads=4,
      winit='trunc_normal_in', binit='zeros'):
    self.units = units
    self.num_seeds = num_seeds
    self.heads = heads
    self.winit = winit
    self.binit = binit

  def __call__(self, tokens, mask, bdims):
    bshape = tokens.shape[:bdims]
    tokens = nn.cast(tokens.reshape((-1, tokens.shape[-2], tokens.shape[-1])))
    mask = mask.reshape((-1, mask.shape[-1])).astype(bool)
    dtype = tokens.dtype
    seeds = self.value(
        'seeds', nn.Initializer('trunc_normal', 'out'),
        (self.num_seeds, self.units), f32).astype(dtype)
    seeds = jnp.broadcast_to(seeds, (tokens.shape[0], self.num_seeds, self.units))
    q = self.sub(
        'q', nn.Linear, self.units,
        winit=self.winit, binit=self.binit)(seeds)
    k = self.sub(
        'k', nn.Linear, self.units,
        winit=self.winit, binit=self.binit)(tokens)
    v = self.sub(
        'v', nn.Linear, self.units,
        winit=self.winit, binit=self.binit)(tokens)
    assert self.units % self.heads == 0, (self.units, self.heads)
    head_dim = self.units // self.heads
    q = q.reshape((-1, self.num_seeds, self.heads, head_dim))
    k = k.reshape((-1, tokens.shape[-2], self.heads, head_dim))
    v = v.reshape((-1, tokens.shape[-2], self.heads, head_dim))
    logits = jnp.einsum('bshd,bnhd->bhsn', q, k)
    logits = f32(logits) / jnp.sqrt(f32(head_dim))
    logits = jnp.where(mask[:, None, None, :], logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
    pooled = jnp.einsum('bhsn,bnhd->bshd', weights, v)
    pooled = pooled.reshape((-1, self.num_seeds, self.units))
    return pooled.reshape((*bshape, self.num_seeds, self.units))


class ActionEncoder(nj.Module):

  def __init__(self, max_issues=10, max_values=10, action_types=2,
      embed_units=32, token_units=128, action_units=256, num_seeds=2,
      heads=4, layers=2, act='silu', norm='rms',
      winit='trunc_normal_in', binit='zeros'):
    self.max_issues = max_issues
    self.max_values = max_values
    self.action_types = action_types
    self.embed_units = embed_units
    self.token_units = token_units
    self.action_units = action_units
    self.num_seeds = num_seeds
    self.heads = heads
    self.layers = layers
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit

  def __call__(self, action_type, offer_values, issue_mask, bdims):
    bshape = offer_values.shape[:bdims]
    offer_values = jnp.clip(offer_values, 0, self.max_values - 1)
    action_type = jnp.clip(action_type, 0, self.action_types - 1)
    issue_ids = jnp.arange(self.max_issues, dtype=jnp.int32)
    issue_ids = jnp.broadcast_to(issue_ids, (*bshape, self.max_issues))
    issue_emb = self.sub(
        'issue_emb', nn.Embed, self.max_issues, self.embed_units,
        shape=(self.max_issues,))(issue_ids)
    value_emb = self.sub(
        'value_emb', nn.Embed, self.max_values, self.embed_units,
        shape=(self.max_issues,))(offer_values)
    type_emb = self.sub(
        'type_emb', nn.Embed, self.action_types, self.embed_units)(action_type)
    type_tokens = jnp.broadcast_to(
        type_emb[..., None, :], (*bshape, self.max_issues, self.embed_units))
    issue_mask_f = issue_mask.astype(f32)[..., None]
    token_inp = jnp.concatenate([
        issue_emb,
        value_emb,
        type_tokens,
        issue_mask_f,
    ], -1)
    tokens = self.sub(
        'token_mlp', nn.MLP, self.layers, self.token_units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(token_inp)
    tokens = tokens * issue_mask_f.astype(tokens.dtype)
    pooled = self.sub(
        'pma', MaskedPMA, self.token_units, self.num_seeds, self.heads,
        winit=self.winit, binit=self.binit)(tokens, issue_mask, bdims)
    flat = pooled.reshape((*bshape, -1))
    context = jnp.concatenate([flat, type_emb], -1)
    emb = self.sub(
        'out', nn.Linear, self.action_units,
        winit=self.winit, binit=self.binit)(nn.cast(context))
    return emb.astype(f32)


class FixedProposalActionEncoder(nj.Module):

  def __init__(self, proposals=None, units=64, layers=2, act='silu',
      norm='rms', winit='trunc_normal_in', binit='zeros'):
    self.proposals = np.asarray(
        fixed_5x6_proposals() if proposals is None else proposals, np.int32)
    self.units = units
    self.layers = layers
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit

  def __call__(self, action, obs, bdims, target_raw=None):
    bshape = action.shape[:bdims]
    n_props = self.proposals.shape[0]
    action = jnp.clip(action.astype(i32), 0, n_props)
    is_accept = action >= n_props
    proposal_id = jnp.minimum(action, n_props - 1)
    proposals = jnp.asarray(self.proposals, f32) / 4.0
    proposal_norm = jnp.take(proposals, proposal_id, axis=0)
    last_opp = jnp.clip(obs['last_opp_offer'], 0, 4).astype(f32) / 4.0
    proposal_norm = jnp.where(is_accept[..., None], last_opp, proposal_norm)
    accept = is_accept.astype(f32)[..., None]
    offer = (1.0 - accept).astype(f32)
    if target_raw is None:
      target_raw = jnp.zeros((*bshape, 4), f32)
    target_raw = sg(f32(target_raw))
    x = jnp.concatenate([proposal_norm, accept, offer, target_raw], -1)
    x = x.reshape((*bshape, -1))
    x = self.sub(
        'mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(nn.cast(x))
    return x.astype(f32)


class IssuePairObsEncoder(nj.Module):

  def __init__(self, max_issues=10, max_values=10, embed_units=32,
      token_units=128, obs_units=512, num_seeds=4, heads=4, layers=2,
      transformer_layers=1, ffup=2, act='silu', norm='rms',
      winit='trunc_normal_in', binit='zeros'):
    self.max_issues = max_issues
    self.max_values = max_values
    self.embed_units = embed_units
    self.token_units = token_units
    self.obs_units = obs_units
    self.num_seeds = num_seeds
    self.heads = heads
    self.layers = layers
    self.transformer_layers = transformer_layers
    self.ffup = ffup
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit

  def __call__(self, obs, bdims, training=False):
    bshape = obs['issue_mask'].shape[:bdims]
    issue_mask = obs['issue_mask'].astype(bool)
    last_opp = jnp.clip(obs['last_opp_offer'], 0, self.max_values - 1)
    last_self = jnp.clip(obs['last_self_offer'], 0, self.max_values - 1)
    issue_ids = jnp.arange(self.max_issues, dtype=jnp.int32)
    issue_ids = jnp.broadcast_to(issue_ids, (*bshape, self.max_issues))
    issue_emb = self.sub(
        'issue_emb', nn.Embed, self.max_issues, self.embed_units,
        shape=(self.max_issues,))(issue_ids)
    opp_emb = self.sub(
        'opp_value_emb', nn.Embed, self.max_values, self.embed_units,
        shape=(self.max_issues,))(last_opp)
    self_emb = self.sub(
        'self_value_emb', nn.Embed, self.max_values, self.embed_units,
        shape=(self.max_issues,))(last_self)
    self_values = obs['self_values']
    self_weights = obs['self_weights']
    opp_local = jnp.take_along_axis(
        self_values, last_opp[..., None], axis=-1).squeeze(-1)
    self_local = jnp.take_along_axis(
        self_values, last_self[..., None], axis=-1).squeeze(-1)
    relative_time = obs['relative_time'][..., None]
    time_tok = jnp.broadcast_to(relative_time[..., None], (*bshape, self.max_issues, 1))
    valid_opp = obs['last_opp_valid'][..., None, None]
    valid_self = obs['last_self_valid'][..., None, None]
    scalars = jnp.stack([
        opp_local,
        self_local,
        self_weights,
        (last_opp == last_self).astype(f32),
        jnp.broadcast_to(obs['last_opp_valid'][..., None], (*bshape, self.max_issues)),
        jnp.broadcast_to(obs['last_self_valid'][..., None], (*bshape, self.max_issues)),
        issue_mask.astype(f32),
    ], -1)
    token_inp = jnp.concatenate([
        issue_emb,
        opp_emb * valid_opp.astype(opp_emb.dtype),
        self_emb * valid_self.astype(self_emb.dtype),
        scalars,
        time_tok,
    ], -1)
    tokens = self.sub(
        'token_mlp', nn.MLP, self.layers, self.token_units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(token_inp)
    tokens = tokens * issue_mask[..., None].astype(tokens.dtype)
    if self.transformer_layers:
      flat = tokens.reshape((-1, self.max_issues, self.token_units))
      flat_mask = issue_mask.reshape((-1, self.max_issues))
      attn_mask = flat_mask[:, :, None] & flat_mask[:, None, :]
      flat = self.sub(
          'transformer', nn.Transformer, units=self.token_units,
          layers=self.transformer_layers, heads=self.heads, ffup=self.ffup,
          act=self.act, norm=self.norm, rope=False,
          winit=self.winit, binit=self.binit)(flat, attn_mask, training=training)
      tokens = flat.reshape((*bshape, self.max_issues, self.token_units))
    pooled = self.sub(
        'pma', MaskedPMA, self.token_units, self.num_seeds, self.heads,
        winit=self.winit, binit=self.binit)(tokens, issue_mask, bdims)
    flat = pooled.reshape((*bshape, -1))
    out = self.sub(
        'out', nn.Linear, self.obs_units,
        winit=self.winit, binit=self.binit)(nn.cast(flat))
    metrics = {
        'anl_obs/emb_abs_mean': jnp.abs(f32(out)).mean(),
    }
    return out.astype(f32), metrics


class OpponentTableOSI(nj.Module):

  def __init__(self, max_issues=10, max_values=10, embed_units=32,
      hidden=256, layers=2, act='silu', norm='rms',
      winit='trunc_normal_in', binit='zeros', outscale=1.0):
    self.max_issues = max_issues
    self.max_values = max_values
    self.embed_units = embed_units
    self.hidden = hidden
    self.layers = layers
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit
    self.outscale = outscale

  def __call__(self, feat, obs, bdims):
    bshape = feat.shape[:bdims]
    feat = feat.reshape((*bshape, -1))
    issue_mask = obs['issue_mask'].astype(bool)
    value_mask = obs['value_mask'].astype(bool)
    issue_ids = jnp.arange(self.max_issues, dtype=jnp.int32)
    issue_ids = jnp.broadcast_to(issue_ids, (*bshape, self.max_issues))
    issue_emb = self.sub(
        'issue_emb', nn.Embed, self.max_issues, self.embed_units,
        shape=(self.max_issues,))(issue_ids)
    feat_issue = jnp.broadcast_to(
        feat[..., None, :], (*bshape, self.max_issues, feat.shape[-1]))
    weight_inp = jnp.concatenate([
        feat_issue,
        issue_emb,
        obs['self_weights'][..., None],
        issue_mask.astype(f32)[..., None],
    ], -1)
    xw = self.sub(
        'weight_mlp', nn.MLP, self.layers, self.hidden,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(weight_inp)
    raw_w = self.sub(
        'weight_out', nn.Linear, 1, outscale=self.outscale,
        winit=self.winit, binit=self.binit)(xw).squeeze(-1)
    raw_w = jnp.where(issue_mask, raw_w, -1e30)
    weights = jax.nn.softmax(raw_w, axis=-1).astype(f32)

    value_ids = jnp.arange(self.max_values, dtype=jnp.int32)
    value_ids = jnp.broadcast_to(
        value_ids, (*bshape, self.max_issues, self.max_values))
    value_emb = self.sub(
        'value_emb', nn.Embed, self.max_values, self.embed_units,
        shape=(self.max_issues, self.max_values))(value_ids)
    feat_value = jnp.broadcast_to(
        feat[..., None, None, :],
        (*bshape, self.max_issues, self.max_values, feat.shape[-1]))
    issue_value_emb = jnp.broadcast_to(
        issue_emb[..., :, None, :],
        (*bshape, self.max_issues, self.max_values, self.embed_units))
    value_inp = jnp.concatenate([
        feat_value,
        issue_value_emb,
        value_emb,
        obs['self_values'][..., None],
        value_mask.astype(f32)[..., None],
    ], -1)
    xv = self.sub(
        'value_mlp', nn.MLP, self.layers, self.hidden,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(value_inp)
    raw_v = self.sub(
        'value_out', nn.Linear, 1, outscale=self.outscale,
        winit=self.winit, binit=self.binit)(xv).squeeze(-1)
    values = jax.nn.sigmoid(raw_v).astype(f32)
    values = jnp.where(value_mask, values, 0.0)
    metrics = {
        'osi_table/weight_mean': weights.mean(),
        'osi_table/value_mean': values.sum() / jnp.maximum(f32(1.0), value_mask.sum()),
    }
    return {'weights': weights, 'values': values}, metrics

  def loss(self, pred, obs):
    issue_mask = obs['issue_mask'].astype(f32)
    value_mask = obs['value_mask'].astype(f32)
    true_w = obs['true_opp_weights']
    true_v = obs['true_opp_values']
    weight_sq = jnp.square(pred['weights'] - true_w) * issue_mask
    value_sq = jnp.square(pred['values'] - true_v) * value_mask
    weight_loss = weight_sq.sum() / jnp.maximum(f32(1.0), issue_mask.sum())
    value_loss = value_sq.sum() / jnp.maximum(f32(1.0), value_mask.sum())
    loss = weight_loss + value_loss
    metrics = {
        'osi_table/weight_loss': weight_loss,
        'osi_table/value_loss': value_loss,
        'osi_table/loss': loss,
    }
    return loss, metrics


class GaussianTableOSIOutput:

  def __init__(
      self, weight_mean_logits, weight_logstd, value_mean_logits,
      value_logstd, reservation_mean_logits, reservation_logstd,
      issue_mask, value_mask, minstd=0.03):
    self.weight_mean_logits = f32(weight_mean_logits)
    self.weight_logstd = jnp.clip(f32(weight_logstd), -5.0, 2.0)
    self.value_mean_logits = f32(value_mean_logits)
    self.value_logstd = jnp.clip(f32(value_logstd), -5.0, 2.0)
    self.reservation_mean_logits = f32(reservation_mean_logits)
    self.reservation_logstd = jnp.clip(f32(reservation_logstd), -5.0, 2.0)
    self.issue_mask = issue_mask.astype(bool)
    self.value_mask = value_mask.astype(bool)
    self.minstd = f32(minstd)

  def weights_mean(self):
    logits = jnp.where(self.issue_mask, self.weight_mean_logits, -1e30)
    return jax.nn.softmax(logits, -1).astype(f32)

  def weights_std(self):
    std = jax.nn.softplus(self.weight_logstd) + self.minstd
    return jnp.where(self.issue_mask, std, 0.0).astype(f32)

  def values_mean(self):
    mean = jax.nn.sigmoid(self.value_mean_logits).astype(f32)
    return jnp.where(self.value_mask, mean, 0.0)

  def values_std(self):
    std = jax.nn.softplus(self.value_logstd) + self.minstd
    return jnp.where(self.value_mask, std, 0.0).astype(f32)

  def reservation_mean(self):
    return jax.nn.sigmoid(self.reservation_mean_logits).astype(f32)

  def reservation_std(self):
    return (jax.nn.softplus(self.reservation_logstd) + self.minstd).astype(f32)

  def pred(self):
    return {
        'weights_mean': self.weights_mean(),
        'weights_std': self.weights_std(),
        'values_mean': self.values_mean(),
        'values_std': self.values_std(),
        'reservation_mean': self.reservation_mean(),
        'reservation_std': self.reservation_std(),
    }

  def _nll(self, target, mean, std):
    var = jnp.square(jnp.maximum(std, self.minstd))
    return 0.5 * (
        jnp.square(target - mean) / var + jnp.log(var) + jnp.log(2.0 * jnp.pi))

  def loss(self, obs):
    issue_mask = self.issue_mask.astype(f32)
    value_mask = self.value_mask.astype(f32)
    weight_nll = self._nll(
        obs['true_opp_weights'], self.weights_mean(), self.weights_std())
    value_nll = self._nll(
        obs['true_opp_values'], self.values_mean(), self.values_std())
    reservation_nll = self._nll(
        obs['true_opp_reserved'][..., None],
        self.reservation_mean(), self.reservation_std()).squeeze(-1)
    weight_loss = (
        (weight_nll * issue_mask).sum() /
        jnp.maximum(f32(1.0), issue_mask.sum()))
    value_loss = (
        (value_nll * value_mask).sum() /
        jnp.maximum(f32(1.0), value_mask.sum()))
    reservation_loss = reservation_nll.mean()
    return weight_loss + value_loss + reservation_loss

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
    reservation_mae = jnp.abs(
        pred['reservation_mean'].squeeze(-1) - obs['true_opp_reserved']).mean()
    return {
        'osi_table/nll': self.loss(obs),
        'osi_table/weight_mae': weight_mae,
        'osi_table/value_mae': value_mae,
        'osi_table/reservation_mae': reservation_mae,
        'osi_table/weight_std': (
            (pred['weights_std'] * issue_mask).sum() /
            jnp.maximum(f32(1.0), issue_mask.sum())),
        'osi_table/value_std': (
            (pred['values_std'] * value_mask).sum() /
            jnp.maximum(f32(1.0), value_mask.sum())),
        'osi_table/reservation_std': pred['reservation_std'].mean(),
    }


class GaussianTableOSIHead(nj.Module):

  def __init__(self, max_issues=6, max_values=5, layers=2, units=256,
      act='silu', norm='rms', minstd=0.03, outscale=1.0,
      winit='trunc_normal_in', binit='zeros'):
    self.max_issues = max_issues
    self.max_values = max_values
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.minstd = minstd
    self.outscale = outscale
    self.winit = winit
    self.binit = binit

  def __call__(self, feat, obs, bdims):
    bshape = feat.shape[:bdims]
    x = feat.reshape((*bshape, -1))
    x = jnp.concatenate([x, obs['relative_time'][..., None]], -1)
    x = self.sub(
        'mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(x)
    kw = dict(winit=self.winit, binit=self.binit, outscale=self.outscale)
    wm = self.sub('weight_mean', nn.Linear, self.max_issues, **kw)(x)
    ws = self.sub('weight_logstd', nn.Linear, self.max_issues, **kw)(x)
    vm = self.sub(
        'value_mean', nn.Linear, (self.max_issues, self.max_values), **kw)(x)
    vs = self.sub(
        'value_logstd', nn.Linear, (self.max_issues, self.max_values), **kw)(x)
    rm = self.sub('reservation_mean', nn.Linear, 1, **kw)(x)
    rs = self.sub('reservation_logstd', nn.Linear, 1, **kw)(x)
    return GaussianTableOSIOutput(
        wm, ws, vm, vs, rm, rs, obs['issue_mask'], obs['value_mask'],
        self.minstd)


def score_offer_from_table(offer_values, weights, values, issue_mask):
  offer_values = jnp.clip(offer_values, 0, values.shape[-1] - 1)
  chosen = jnp.take_along_axis(values, offer_values[..., None], axis=-1)
  chosen = chosen.squeeze(-1)
  return (weights * chosen * issue_mask.astype(f32)).sum(-1)


def score_proposals_from_table(proposals, weights, values, issue_mask):
  chosen = gather_table_values(values, proposals)
  return (
      weights[..., None, :] * chosen *
      issue_mask.astype(f32)[..., None, :]).sum(-1)


def proposal_mask_from_obs(proposals, obs):
  proposals = jnp.asarray(proposals, jnp.int32)
  value_counts = obs['value_mask'].astype(jnp.int32).sum(-1)
  active = obs['issue_mask'].astype(bool)
  p = proposals.reshape((*(1,) * (active.ndim - 1), *proposals.shape))
  p = p + jnp.zeros((*active.shape[:-1], *proposals.shape), jnp.int32)
  counts = value_counts[..., None, :]
  active_i = active[..., None, :]
  valid_active = p < counts
  valid_inactive = p == 0
  return jnp.where(active_i, valid_active, valid_inactive).all(-1)


def gather_table_values(table, proposals):
  proposals = jnp.asarray(proposals, jnp.int32)
  p = proposals.reshape((*(1,) * (table.ndim - 2), *proposals.shape))
  p = p + jnp.zeros((*table.shape[:-2], *proposals.shape), jnp.int32)
  table_exp = table[..., None, :, :]
  idx = p[..., :, None]
  return jnp.take_along_axis(table_exp, idx, axis=-1).squeeze(-1)


def proposal_utility_stats(proposals, weight_mean, weight_std, value_mean,
    value_std, reservation_mean, reservation_std, issue_mask):
  issue_mask_f = issue_mask.astype(f32)[..., None, :]
  vmean = gather_table_values(value_mean, proposals)
  vstd = gather_table_values(value_std, proposals)
  wmean = weight_mean[..., None, :]
  wstd = weight_std[..., None, :]
  mean = (wmean * vmean * issue_mask_f).sum(-1)
  var_terms = (
      jnp.square(wstd) * jnp.square(vstd) +
      jnp.square(wstd) * jnp.square(vmean) +
      jnp.square(vstd) * jnp.square(wmean))
  var = (var_terms * issue_mask_f).sum(-1)
  std = jnp.sqrt(jnp.maximum(var, 1e-6))
  margin_mean = mean - reservation_mean
  margin_std = jnp.sqrt(jnp.square(std) + jnp.square(reservation_std))
  return mean, std, margin_mean, margin_std


class FixedProposalLandscapeEncoder(nj.Module):

  def __init__(self, proposals=None, d_model=32, num_seeds=4, heads=4,
      token_layers=1, latent_units=256, act='silu', norm='rms',
      winit='trunc_normal_in', binit='zeros', outscale=1.0):
    self.proposals = np.asarray(
        fixed_5x6_proposals() if proposals is None else proposals, np.int32)
    self.d_model = d_model
    self.num_seeds = num_seeds
    self.heads = heads
    self.token_layers = token_layers
    self.latent_units = latent_units
    self.act = act
    self.norm = norm
    self.winit = winit
    self.binit = binit
    self.outscale = outscale

  def __call__(self, obs, osi_pred, bdims):
    bshape = obs['issue_mask'].shape[:bdims]
    proposals = jnp.asarray(self.proposals, jnp.int32)
    proposal_mask = proposal_mask_from_obs(proposals, obs)
    proposal_norm = f32(proposals) / 4.0
    proposal_norm = proposal_norm.reshape(
        (*(1,) * bdims, *proposal_norm.shape))
    proposal_norm = proposal_norm + jnp.zeros(
        (*bshape, self.proposals.shape[0], self.proposals.shape[1]), f32)
    self_u = score_proposals_from_table(
        proposals, obs['self_weights'], obs['self_values'], obs['issue_mask'])
    self_margin = self_u - obs['self_reserved'][..., None]
    opp_u, opp_std, opp_margin, opp_margin_std = proposal_utility_stats(
        proposals,
        osi_pred['weights_mean'], osi_pred['weights_std'],
        osi_pred['values_mean'], osi_pred['values_std'],
        osi_pred['reservation_mean'], osi_pred['reservation_std'],
        obs['issue_mask'])
    tokens = jnp.concatenate([
        proposal_norm,
        self_u[..., None],
        self_margin[..., None],
        opp_u[..., None],
        opp_std[..., None],
        opp_margin[..., None],
        opp_margin_std[..., None],
        (self_u - opp_u)[..., None],
        (self_u + opp_u)[..., None],
        (self_u * opp_u)[..., None],
        proposal_mask.astype(f32)[..., None],
    ], -1)
    tokens = jnp.where(proposal_mask[..., None], tokens, 0.0)
    embeds = self.sub(
        'token_mlp', nn.MLP, self.token_layers, self.d_model,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(tokens)
    pooled = self.sub(
        'pma', MaskedPMA, self.d_model, self.num_seeds, self.heads,
        winit=self.winit, binit=self.binit)(embeds, proposal_mask, bdims)
    flat = pooled.reshape((*bshape, -1))
    landscape = self.sub(
        'compress', nn.Linear, self.latent_units,
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(nn.cast(flat))
    metrics = {
        'landscape/proposal_valid_rate': proposal_mask.mean(),
        'landscape/latent_abs_mean': jnp.abs(f32(landscape)).mean(),
    }
    aux = {
        'proposal_mask': proposal_mask,
        'self_u': self_u,
        'opp_u_mean': opp_u,
        'opp_u_std': opp_std,
        'opp_margin_mean': opp_margin,
        'opp_margin_std': opp_margin_std,
    }
    return landscape.astype(f32), aux, metrics


def _scaled_sigmoid(z, lower, upper):
  return lower + (upper - lower) * jax.nn.sigmoid(z)


class FixedUtilityBandOutput(outs.Output):

  def __init__(self, logits, det_logits, rho, delta, delta_down, z_e,
      z_delta, z_delta_down, z_umax, mu_e, logstd_e, mu_delta,
      logstd_delta, mu_delta_down, logstd_delta_down, mu_umax, logstd_umax,
      reservation, u_max, det_rho, det_delta, det_delta_down, det_u_max,
      u_max_min=0.9, u_max_max=1.0, unimix=0.01):
    self.action = outs.Categorical(logits, unimix)
    self.det_action = outs.Categorical(det_logits, unimix)
    self.rho = f32(rho)
    self.delta = f32(delta)
    self.delta_down = f32(delta_down)
    self.z_e = f32(z_e)
    self.z_delta = f32(z_delta)
    self.z_delta_down = f32(z_delta_down)
    self.z_umax = f32(z_umax)
    self.mu_e = f32(mu_e)
    self.logstd_e = jnp.clip(f32(logstd_e), -5.0, 2.0)
    self.mu_delta = f32(mu_delta)
    self.logstd_delta = jnp.clip(f32(logstd_delta), -5.0, 2.0)
    self.mu_delta_down = f32(mu_delta_down)
    self.logstd_delta_down = jnp.clip(f32(logstd_delta_down), -5.0, 2.0)
    self.mu_umax = f32(mu_umax)
    self.logstd_umax = jnp.clip(f32(logstd_umax), -5.0, 2.0)
    self.reservation = f32(reservation)
    self.u_max = f32(u_max)
    self.det_rho = f32(det_rho)
    self.det_delta = f32(det_delta)
    self.det_delta_down = f32(det_delta_down)
    self.det_u_max = f32(det_u_max)
    self.u_max_min = f32(u_max_min)
    self.u_max_max = f32(u_max_max)
    self.minent = 0
    self.maxent = np.log(logits.shape[-1])

  def pred(self):
    return self.det_action.pred()

  def sample(self, seed, shape=()):
    return self.action.sample(seed, shape)

  def logp(self, event):
    return self.band_logp() + self.action_logp(event)

  def band_logp(self):
    return self._logp_band()

  def action_logp(self, event):
    return self.action.logp(event)

  def entropy(self):
    e_std = jnp.exp(self.logstd_e)
    delta_std = jnp.exp(self.logstd_delta)
    delta_down_std = jnp.exp(self.logstd_delta_down)
    umax_std = jnp.exp(self.logstd_umax)
    eps = f32(1e-6)
    down_max = jnp.minimum(f32(0.1), jnp.maximum(self.rho - self.reservation, 0.0))
    logdet_delta_down = (
        jnp.log(jnp.maximum(down_max, eps)) +
        jax.nn.log_sigmoid(self.z_delta_down) +
        jax.nn.log_sigmoid(-self.z_delta_down))
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    band_ent = (
        0.5 * jnp.log(2 * jnp.pi * jnp.square(e_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(delta_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(delta_down_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(umax_std)) + 0.5 +
        logdet_delta_down +
        logdet_umax)
    return band_ent.squeeze(-1) + self.action.entropy()

  def kl(self, other):
    raise NotImplementedError(type(other))

  def _logp_band(self):
    eps = f32(1e-6)
    e_std = jnp.exp(self.logstd_e)
    delta_std = jnp.exp(self.logstd_delta)
    delta_down_std = jnp.exp(self.logstd_delta_down)
    umax_std = jnp.exp(self.logstd_umax)
    logp_e = -0.5 * (
        jnp.square((self.z_e - self.mu_e) / e_std) +
        2.0 * self.logstd_e + jnp.log(2.0 * jnp.pi))
    logp_delta = -0.5 * (
        jnp.square((self.z_delta - self.mu_delta) / delta_std) +
        2.0 * self.logstd_delta + jnp.log(2.0 * jnp.pi))
    logp_delta_down = -0.5 * (
        jnp.square((self.z_delta_down - self.mu_delta_down) / delta_down_std) +
        2.0 * self.logstd_delta_down + jnp.log(2.0 * jnp.pi))
    logp_umax = -0.5 * (
        jnp.square((self.z_umax - self.mu_umax) / umax_std) +
        2.0 * self.logstd_umax + jnp.log(2.0 * jnp.pi))
    log_e_min = f32(np.log(2.0))
    log_e_max = f32(np.log(50.0))
    log_e_range = log_e_max - log_e_min
    eta = log_e_min + jax.nn.sigmoid(self.z_e) * log_e_range
    logdet_e = (
        eta + jnp.log(jnp.maximum(log_e_range, eps)) +
        jax.nn.log_sigmoid(self.z_e) +
        jax.nn.log_sigmoid(-self.z_e))
    logdet_delta = (
        jnp.log(jnp.maximum(1.0 - sg(self.rho), eps)) +
        jax.nn.log_sigmoid(self.z_delta) +
        jax.nn.log_sigmoid(-self.z_delta))
    down_max = jnp.minimum(
        f32(0.1), jnp.maximum(sg(self.rho) - sg(self.reservation), 0.0))
    logdet_delta_down = (
        jnp.log(jnp.maximum(down_max, eps)) +
        jax.nn.log_sigmoid(self.z_delta_down) +
        jax.nn.log_sigmoid(-self.z_delta_down))
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    return (
        logp_e - logdet_e +
        logp_delta - logdet_delta +
        logp_delta_down - logdet_delta_down +
        logp_umax - logdet_umax).squeeze(-1)


class FixedUtilityBandPolicyHead(nj.Module):

  def __init__(self, proposals=None, layers=3, units=1024, act='silu',
      norm='rms', outscale=0.01, unimix=0.01, winit='trunc_normal_in',
      binit='zeros', e_min=2.0, e_max=50.0, minstd=None, maxstd=None,
      rho_delay_coef=0.0, rho_delay_power=1.0, offer_delay_coef=0.0,
      accept_delay_coef=0.0, delay_power=2.0, u_max_min=0.9,
      u_max_max=1.0):
    self.proposals = np.asarray(
        fixed_5x6_proposals() if proposals is None else proposals, np.int32)
    self.layers = layers
    self.units = units
    self.act = act
    self.norm = norm
    self.outscale = outscale
    self.unimix = unimix
    self.winit = winit
    self.binit = binit
    self.e_min = e_min
    self.e_max = e_max
    self.u_max_min = float(u_max_min)
    self.u_max_max = float(u_max_max)
    self.rho_delay_coef = float(rho_delay_coef)
    self.rho_delay_power = float(rho_delay_power)
    self.offer_delay_coef = float(offer_delay_coef)
    self.accept_delay_coef = float(accept_delay_coef)
    self.delay_power = float(delay_power)

  def __call__(self, head, obs, proposal_aux, bdims):
    bshape = head.shape[:bdims]
    head = nn.cast(head.reshape((*bshape, -1)))
    band = self.sub(
        'band_mlp', nn.MLP, 2, self.units,
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
    mu_delta_down = self.sub(
        'mu_delta_down', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_delta_down = self.sub(
        'logstd_delta_down', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    mu_umax = self.sub(
        'mu_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_umax = self.sub(
        'logstd_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_e = jnp.clip(logstd_e, -5.0, 2.0)
    logstd_delta = jnp.clip(logstd_delta, -5.0, 2.0)
    logstd_delta_down = jnp.clip(logstd_delta_down, -5.0, 2.0)
    logstd_umax = jnp.clip(logstd_umax, -5.0, 2.0)
    key_e, key_delta, key_delta_down, key_umax = jax.random.split(nj.seed(), 4)
    z_e = mu_e + jnp.exp(logstd_e) * jax.random.normal(key_e, mu_e.shape)
    z_delta = (
        mu_delta + jnp.exp(logstd_delta) *
        jax.random.normal(key_delta, mu_delta.shape))
    z_delta_down = (
        mu_delta_down + jnp.exp(logstd_delta_down) *
        jax.random.normal(key_delta_down, mu_delta_down.shape))
    z_umax = (
        mu_umax + jnp.exp(logstd_umax) *
        jax.random.normal(key_umax, mu_umax.shape))
    reservation = jnp.clip(obs['self_reserved'][..., None], 0.0, 1.0)
    time = jnp.clip(obs['relative_time'][..., None], 1e-6, 1.0)
    e = self._concession_e(z_e)
    det_e = self._concession_e(mu_e)
    u_max = self._u_max(z_umax)
    det_u_max = self._u_max(mu_umax)
    rho = reservation + (u_max - reservation) * (1.0 - jnp.power(time, e))
    det_rho = reservation + (det_u_max - reservation) * (
        1.0 - jnp.power(time, det_e))
    rho = jnp.clip(rho, 0.0, 1.0)
    det_rho = jnp.clip(det_rho, 0.0, 1.0)
    delta = (1.0 - rho) * jax.nn.sigmoid(z_delta)
    det_delta = (1.0 - det_rho) * jax.nn.sigmoid(mu_delta)
    down_max = jnp.minimum(f32(0.1), jnp.maximum(rho - reservation, 0.0))
    det_down_max = jnp.minimum(
        f32(0.1), jnp.maximum(det_rho - reservation, 0.0))
    delta_down = down_max * jax.nn.sigmoid(z_delta_down)
    det_delta_down = det_down_max * jax.nn.sigmoid(mu_delta_down)
    logits = self._action_logits(
        head, obs, proposal_aux, rho, delta, delta_down)
    det_logits = self._action_logits(
        head, obs, proposal_aux, det_rho, det_delta, det_delta_down)
    return {'action': FixedUtilityBandOutput(
        logits, det_logits, rho, delta, delta_down, z_e, z_delta,
        z_delta_down, z_umax, mu_e, logstd_e, mu_delta, logstd_delta,
        mu_delta_down, logstd_delta_down, mu_umax, logstd_umax,
        reservation, u_max, det_rho, det_delta, det_delta_down, det_u_max,
        self.u_max_min, self.u_max_max, self.unimix)}

  def _concession_e(self, z_e):
    log_e_min = f32(np.log(self.e_min))
    log_e_max = f32(np.log(self.e_max))
    eta = log_e_min + jax.nn.sigmoid(z_e) * (log_e_max - log_e_min)
    return jnp.exp(eta)

  def _u_max(self, z_umax):
    return _scaled_sigmoid(z_umax, f32(self.u_max_min), f32(self.u_max_max))

  def _action_logits(self, head, obs, proposal_aux, rho, delta, delta_down):
    x = self.sub(
        'action_mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(
            jnp.concatenate([head, rho, delta_down, delta], -1))
    n_props = self.proposals.shape[0]
    proposal_logits = self.sub(
        'proposal_logits', nn.Linear, n_props,
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(x)
    accept_logit = self.sub(
        'accept_logit', nn.Linear, 1,
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(x)
    self_u = proposal_aux['self_u']
    proposal_mask = proposal_aux['proposal_mask']
    lower = jnp.maximum(obs['self_reserved'][..., None], rho - delta_down)
    in_band = jnp.logical_and(self_u >= lower, self_u <= rho + delta)
    valid = jnp.logical_and(proposal_mask, in_band)
    nearest_distance = jnp.where(
        proposal_mask, jnp.abs(self_u - rho), jnp.inf)
    nearest = jnp.argmin(nearest_distance, -1)
    fallback = jax.nn.one_hot(nearest, n_props).astype(bool)
    has_valid = valid.any(-1, keepdims=True)
    proposal_valid = jnp.where(has_valid, valid, fallback)
    masked_proposal_logits = jnp.where(
        proposal_valid, proposal_logits, jnp.full_like(proposal_logits, -1e9))
    accept_allowed = self._accept_allowed(obs, rho)
    accept_logit = jnp.where(
        accept_allowed, accept_logit, jnp.full_like(accept_logit, -1e9))
    return jnp.concatenate([masked_proposal_logits, accept_logit], -1)

  def _accept_allowed(self, obs, rho):
    last_opp = obs['last_opp_offer']
    last_opp = jnp.clip(last_opp, 0, obs['self_values'].shape[-1] - 1)
    self_u = score_offer_from_table(
        last_opp, obs['self_weights'], obs['self_values'], obs['issue_mask'])
    return jnp.logical_and(obs['last_opp_valid'][..., None], self_u[..., None] >= rho)


class FixedCandidateUtilityBandOutput(outs.Output):

  def __init__(self, logits, det_logits, candidate_idx, det_candidate_idx,
      candidate_mask, det_candidate_mask, action_mask, det_action_mask,
      accept_id, rho, delta, delta_down, z_e, z_delta, z_delta_down, mu_e,
      logstd_e, mu_delta, logstd_delta, mu_delta_down, logstd_delta_down,
      z_umax, mu_umax, logstd_umax, reservation, u_max, det_rho, det_delta,
      det_delta_down, det_u_max, u_max_min=0.9, u_max_max=1.0, unimix=0.01):
    self.logits = self._masked_logits(logits, action_mask, unimix)
    self.det_logits = self._masked_logits(det_logits, det_action_mask, unimix)
    self.candidate_idx = candidate_idx.astype(i32)
    self.det_candidate_idx = det_candidate_idx.astype(i32)
    self.candidate_mask = candidate_mask.astype(bool)
    self.det_candidate_mask = det_candidate_mask.astype(bool)
    self.action_mask = action_mask.astype(bool)
    self.det_action_mask = det_action_mask.astype(bool)
    self.accept_id = i32(accept_id)
    self.rho = f32(rho)
    self.delta = f32(delta)
    self.delta_down = f32(delta_down)
    self.z_e = f32(z_e)
    self.z_delta = f32(z_delta)
    self.z_delta_down = f32(z_delta_down)
    self.z_umax = f32(z_umax)
    self.mu_e = f32(mu_e)
    self.logstd_e = jnp.clip(f32(logstd_e), -5.0, 2.0)
    self.mu_delta = f32(mu_delta)
    self.logstd_delta = jnp.clip(f32(logstd_delta), -5.0, 2.0)
    self.mu_delta_down = f32(mu_delta_down)
    self.logstd_delta_down = jnp.clip(f32(logstd_delta_down), -5.0, 2.0)
    self.mu_umax = f32(mu_umax)
    self.logstd_umax = jnp.clip(f32(logstd_umax), -5.0, 2.0)
    self.reservation = f32(reservation)
    self.u_max = f32(u_max)
    self.det_rho = f32(det_rho)
    self.det_delta = f32(det_delta)
    self.det_delta_down = f32(det_delta_down)
    self.det_u_max = f32(det_u_max)
    self.u_max_min = f32(u_max_min)
    self.u_max_max = f32(u_max_max)
    self.minent = 0
    self.maxent = np.log(logits.shape[-1])

  def pred(self):
    internal = jnp.argmax(self.det_logits, -1)
    return self._map_internal(internal, self.det_candidate_idx)

  def sample(self, seed, shape=()):
    internal = jax.random.categorical(
        seed, self.logits, -1, shape + self.logits.shape[:-1])
    return self._map_internal(internal, self.candidate_idx)

  def logp(self, event):
    return self.band_logp() + self.action_logp(event)

  def band_logp(self):
    return self._logp_band()

  def action_logp(self, event):
    event = event.astype(i32)
    logprob = jax.nn.log_softmax(self.logits, -1)
    proposal_logprob = logprob[..., :-1]
    accept_logprob = logprob[..., -1]
    match = jnp.equal(self.candidate_idx, event[..., None])
    matched = jnp.logical_and(match, self.candidate_mask)
    proposal = jax.nn.logsumexp(
        jnp.where(matched, proposal_logprob, -1e9), -1)
    accept = jnp.where(
        event == self.accept_id, accept_logprob, jnp.full_like(proposal, -1e9))
    return jnp.maximum(proposal, accept)

  def entropy(self):
    e_std = jnp.exp(self.logstd_e)
    delta_std = jnp.exp(self.logstd_delta)
    delta_down_std = jnp.exp(self.logstd_delta_down)
    umax_std = jnp.exp(self.logstd_umax)
    eps = f32(1e-6)
    down_max = jnp.minimum(f32(0.1), jnp.maximum(self.rho - self.reservation, 0.0))
    logdet_delta_down = (
        jnp.log(jnp.maximum(down_max, eps)) +
        jax.nn.log_sigmoid(self.z_delta_down) +
        jax.nn.log_sigmoid(-self.z_delta_down))
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    band_ent = (
        0.5 * jnp.log(2 * jnp.pi * jnp.square(e_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(delta_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(delta_down_std)) + 0.5 +
        0.5 * jnp.log(2 * jnp.pi * jnp.square(umax_std)) + 0.5 +
        logdet_delta_down +
        logdet_umax)
    logprob = jax.nn.log_softmax(self.logits, -1)
    prob = jax.nn.softmax(self.logits, -1)
    return band_ent.squeeze(-1) - (prob * logprob).sum(-1)

  def kl(self, other):
    raise NotImplementedError(type(other))

  def _map_internal(self, internal, candidate_idx):
    m = candidate_idx.shape[-1]
    safe_internal = jnp.minimum(internal, m - 1)
    proposal = jnp.take_along_axis(
        candidate_idx, safe_internal[..., None], -1).squeeze(-1)
    return jnp.where(internal == m, self.accept_id, proposal).astype(i32)

  def _masked_logits(self, logits, mask, unimix):
    logits = jnp.where(mask, logits, jnp.full_like(logits, -1e9))
    if not unimix:
      return f32(logits)
    probs = jax.nn.softmax(logits, -1)
    denom = jnp.maximum(mask.astype(f32).sum(-1, keepdims=True), 1.0)
    uniform = mask.astype(f32) / denom
    probs = (1 - unimix) * probs + unimix * uniform
    return jnp.where(mask, jnp.log(jnp.maximum(probs, 1e-30)), -1e9)

  def _logp_band(self):
    eps = f32(1e-6)
    e_std = jnp.exp(self.logstd_e)
    delta_std = jnp.exp(self.logstd_delta)
    delta_down_std = jnp.exp(self.logstd_delta_down)
    umax_std = jnp.exp(self.logstd_umax)
    logp_e = -0.5 * (
        jnp.square((self.z_e - self.mu_e) / e_std) +
        2.0 * self.logstd_e + jnp.log(2.0 * jnp.pi))
    logp_delta = -0.5 * (
        jnp.square((self.z_delta - self.mu_delta) / delta_std) +
        2.0 * self.logstd_delta + jnp.log(2.0 * jnp.pi))
    logp_delta_down = -0.5 * (
        jnp.square((self.z_delta_down - self.mu_delta_down) / delta_down_std) +
        2.0 * self.logstd_delta_down + jnp.log(2.0 * jnp.pi))
    logp_umax = -0.5 * (
        jnp.square((self.z_umax - self.mu_umax) / umax_std) +
        2.0 * self.logstd_umax + jnp.log(2.0 * jnp.pi))
    log_e_min = f32(np.log(2.0))
    log_e_max = f32(np.log(50.0))
    log_e_range = log_e_max - log_e_min
    eta = log_e_min + jax.nn.sigmoid(self.z_e) * log_e_range
    logdet_e = (
        eta + jnp.log(jnp.maximum(log_e_range, eps)) +
        jax.nn.log_sigmoid(self.z_e) +
        jax.nn.log_sigmoid(-self.z_e))
    logdet_delta = (
        jnp.log(jnp.maximum(1.0 - sg(self.rho), eps)) +
        jax.nn.log_sigmoid(self.z_delta) +
        jax.nn.log_sigmoid(-self.z_delta))
    down_max = jnp.minimum(
        f32(0.1), jnp.maximum(sg(self.rho) - sg(self.reservation), 0.0))
    logdet_delta_down = (
        jnp.log(jnp.maximum(down_max, eps)) +
        jax.nn.log_sigmoid(self.z_delta_down) +
        jax.nn.log_sigmoid(-self.z_delta_down))
    u_range = self.u_max_max - self.u_max_min
    logdet_umax = (
        jnp.log(jnp.maximum(u_range, eps)) +
        jax.nn.log_sigmoid(self.z_umax) +
        jax.nn.log_sigmoid(-self.z_umax))
    return (
        logp_e - logdet_e +
        logp_delta - logdet_delta +
        logp_delta_down - logdet_delta_down +
        logp_umax - logdet_umax).squeeze(-1)


class FixedCandidateUtilityBandPolicyHead(FixedUtilityBandPolicyHead):

  def __init__(self, proposals=None, candidates=256, candidate_units=200,
      candidate_layers=2, rho_delay_coef=0.0, rho_delay_power=1.0,
      offer_delay_coef=0.0, accept_delay_coef=0.0, delay_power=2.0, **kwargs):
    super().__init__(
        proposals=proposals, rho_delay_coef=rho_delay_coef,
        rho_delay_power=rho_delay_power, offer_delay_coef=offer_delay_coef,
        accept_delay_coef=accept_delay_coef, delay_power=delay_power, **kwargs)
    self.candidates = candidates
    self.candidate_units = candidate_units
    self.candidate_layers = candidate_layers

  def __call__(self, head, obs, proposal_aux, bdims):
    bshape = head.shape[:bdims]
    head = nn.cast(head.reshape((*bshape, -1)))
    band = self.sub(
        'band_mlp', nn.MLP, 2, self.units,
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
    mu_delta_down = self.sub(
        'mu_delta_down', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_delta_down = self.sub(
        'logstd_delta_down', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    mu_umax = self.sub(
        'mu_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_umax = self.sub(
        'logstd_umax', nn.Linear, 1,
        winit=self.winit, binit=self.binit, outscale=self.outscale)(band)
    logstd_e = jnp.clip(logstd_e, -5.0, 2.0)
    logstd_delta = jnp.clip(logstd_delta, -5.0, 2.0)
    logstd_delta_down = jnp.clip(logstd_delta_down, -5.0, 2.0)
    logstd_umax = jnp.clip(logstd_umax, -5.0, 2.0)
    key_e, key_delta, key_delta_down, key_umax = jax.random.split(nj.seed(), 4)
    z_e = mu_e + jnp.exp(logstd_e) * jax.random.normal(key_e, mu_e.shape)
    z_delta = (
        mu_delta + jnp.exp(logstd_delta) *
        jax.random.normal(key_delta, mu_delta.shape))
    z_delta_down = (
        mu_delta_down + jnp.exp(logstd_delta_down) *
        jax.random.normal(key_delta_down, mu_delta_down.shape))
    z_umax = (
        mu_umax + jnp.exp(logstd_umax) *
        jax.random.normal(key_umax, mu_umax.shape))
    reservation = jnp.clip(obs['self_reserved'][..., None], 0.0, 1.0)
    raw_time = jnp.clip(obs['relative_time'][..., None], 0.0, 1.0)
    time = jnp.clip(raw_time, 1e-6, 1.0)
    e = self._concession_e(z_e)
    det_e = self._concession_e(mu_e)
    u_max = self._u_max(z_umax)
    det_u_max = self._u_max(mu_umax)
    rho = reservation + (u_max - reservation) * (1.0 - jnp.power(time, e))
    det_rho = reservation + (det_u_max - reservation) * (
        1.0 - jnp.power(time, det_e))
    rho = jnp.clip(rho, 0.0, 1.0)
    det_rho = jnp.clip(det_rho, 0.0, 1.0)
    rho = self._delay_rho(rho, raw_time)
    det_rho = self._delay_rho(det_rho, raw_time)
    delta = (1.0 - rho) * jax.nn.sigmoid(z_delta)
    det_delta = (1.0 - det_rho) * jax.nn.sigmoid(mu_delta)
    down_max = jnp.minimum(f32(0.1), jnp.maximum(rho - reservation, 0.0))
    det_down_max = jnp.minimum(
        f32(0.1), jnp.maximum(det_rho - reservation, 0.0))
    delta_down = down_max * jax.nn.sigmoid(z_delta_down)
    det_delta_down = det_down_max * jax.nn.sigmoid(mu_delta_down)
    logits, candidate_idx, candidate_mask, action_mask = self._candidate_logits(
        head, obs, proposal_aux, rho, delta, delta_down, u_max,
        stochastic=True)
    det_logits, det_candidate_idx, det_candidate_mask, det_action_mask = (
        self._candidate_logits(
            head, obs, proposal_aux, det_rho, det_delta, det_delta_down,
            det_u_max, stochastic=False))
    accept_id = self.proposals.shape[0]
    return {'action': FixedCandidateUtilityBandOutput(
        logits, det_logits, candidate_idx, det_candidate_idx,
        candidate_mask, det_candidate_mask, action_mask, det_action_mask,
        accept_id, rho, delta, delta_down, z_e, z_delta, z_delta_down,
        mu_e, logstd_e, mu_delta, logstd_delta, mu_delta_down,
        logstd_delta_down, z_umax, mu_umax, logstd_umax, reservation, u_max,
        det_rho, det_delta, det_delta_down, det_u_max, self.u_max_min,
        self.u_max_max, self.unimix)}

  def _delay_rho(self, rho, time):
    if self.rho_delay_coef <= 0.0:
      return rho
    gate = jnp.power(1.0 - time, self.rho_delay_power)
    rho = rho + self.rho_delay_coef * gate * (1.0 - rho)
    return jnp.clip(rho, 0.0, 1.0)

  def _candidate_logits(
      self, head, obs, proposal_aux, rho, delta, delta_down, u_max,
      stochastic):
    candidate_idx, candidate_mask = self._select_candidates(
        proposal_aux, obs, rho, delta, delta_down, stochastic)
    tokens = self._candidate_tokens(obs, proposal_aux, candidate_idx)
    head_rep = jnp.broadcast_to(
        head[..., None, :], (*head.shape[:-1], self.candidates, head.shape[-1]))
    x = jnp.concatenate([
        head_rep,
        tokens,
        jnp.broadcast_to(rho[..., None, :], (*tokens.shape[:-1], 1)),
        jnp.broadcast_to(delta_down[..., None, :], (*tokens.shape[:-1], 1)),
        jnp.broadcast_to(delta[..., None, :], (*tokens.shape[:-1], 1)),
        jnp.broadcast_to(u_max[..., None, :], (*tokens.shape[:-1], 1)),
    ], -1)
    x = self.sub(
        'candidate_mlp', nn.MLP, self.candidate_layers, self.candidate_units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(x)
    proposal_logits = self.sub(
        'candidate_score', nn.Linear, 1,
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(x).squeeze(-1)
    last_opp_u, last_opp_margin = self._last_opp_self_features(obs)
    accept_features = jnp.concatenate([
        head, rho, delta_down, delta, u_max, last_opp_u, last_opp_margin], -1)
    accept_inp = self.sub(
        'accept_mlp', nn.MLP, self.layers, self.units,
        act=self.act, norm=self.norm,
        winit=self.winit, binit=self.binit)(
            accept_features)
    accept_logit = self.sub(
        'accept_logit', nn.Linear, 1,
        winit=self.winit, binit=self.binit,
        outscale=self.outscale)(accept_inp)
    accept_allowed = self._accept_allowed(obs, rho)
    action_mask = jnp.concatenate([candidate_mask, accept_allowed], -1)
    logits = jnp.concatenate([proposal_logits, accept_logit], -1)
    return logits, candidate_idx, candidate_mask, action_mask

  def _last_opp_self_features(self, obs):
    last_opp = obs['last_opp_offer']
    last_opp = jnp.clip(last_opp, 0, obs['self_values'].shape[-1] - 1)
    self_u = score_offer_from_table(
        last_opp, obs['self_weights'], obs['self_values'], obs['issue_mask'])
    valid = obs['last_opp_valid'].astype(f32)
    self_u = jnp.where(valid, self_u, jnp.zeros_like(self_u))
    margin = jnp.where(
        valid, self_u - obs['self_reserved'], jnp.zeros_like(self_u))
    return self_u[..., None], margin[..., None]

  def _select_candidates(self, proposal_aux, obs, rho, delta, delta_down,
      stochastic):
    n_props = self.proposals.shape[0]
    m = min(self.candidates, n_props)
    self_u = proposal_aux['self_u']
    proposal_mask = proposal_aux['proposal_mask']
    lower = jnp.maximum(obs['self_reserved'][..., None], rho - delta_down)
    in_band = jnp.logical_and(self_u >= lower, self_u <= rho + delta)
    valid = jnp.logical_and(proposal_mask, in_band)
    nearest_distance = jnp.where(proposal_mask, jnp.abs(self_u - rho), jnp.inf)
    nearest = jnp.argmin(nearest_distance, -1)
    fallback = jax.nn.one_hot(nearest, n_props).astype(bool)
    has_valid = valid.any(-1, keepdims=True)
    selector = jnp.where(has_valid, valid, fallback)
    # Match train/eval candidate coverage: sample the candidate set from all
    # valid band offers, while the final action is still chosen by logits.
    gumbel = jax.random.gumbel(nj.seed(), self_u.shape)
    scores = jnp.where(selector, gumbel, -1e9)
    top_scores, idx = jax.lax.top_k(scores, m)
    mask = top_scores > -1e8
    if m < self.candidates:
      pad = self.candidates - m
      idx = jnp.pad(idx, [(0, 0)] * (idx.ndim - 1) + [(0, pad)])
      mask = jnp.pad(mask, [(0, 0)] * (mask.ndim - 1) + [(0, pad)])
    return idx.astype(i32), mask.astype(bool)

  def _candidate_tokens(self, obs, proposal_aux, candidate_idx):
    proposals = jnp.asarray(self.proposals, f32) / 4.0
    proposal_norm = jnp.take(proposals, candidate_idx, axis=0)
    gather = lambda x: jnp.take_along_axis(x, candidate_idx, -1)
    self_u = gather(proposal_aux['self_u'])
    opp_u = gather(proposal_aux['opp_u_mean'])
    opp_std = gather(proposal_aux['opp_u_std'])
    opp_margin = gather(proposal_aux['opp_margin_mean'])
    opp_margin_std = gather(proposal_aux['opp_margin_std'])
    self_margin = self_u - obs['self_reserved'][..., None]
    return jnp.concatenate([
        proposal_norm,
        self_u[..., None],
        self_margin[..., None],
        opp_u[..., None],
        opp_std[..., None],
        opp_margin[..., None],
        opp_margin_std[..., None],
        (self_u - opp_u)[..., None],
        (self_u + opp_u)[..., None],
        (self_u * opp_u)[..., None],
    ], -1).astype(f32)


class ANLDreamerCore(nj.Module):

  def __init__(self, dyn_act_space, obs_encoder=None, action_encoder=None,
      osi=None, dyn=None, action_units=256, max_issues=10, max_values=10,
      act='silu', norm='rms', winit='trunc_normal_in'):
    self.dyn_act_space = dyn_act_space
    self.obs_encoder_kw = obs_encoder or {}
    self.action_encoder_kw = action_encoder or {}
    self.osi_kw = osi or {}
    self.dyn_kw = dyn or {}
    self.action_units = action_units
    self.max_issues = max_issues
    self.max_values = max_values
    self.act = act
    self.norm = norm
    self.winit = winit

  def initial(self, batch_size):
    return self.sub(
        'dyn', rssm.RSSM, self.dyn_act_space, **self.dyn_kw).initial(batch_size)

  def encode_obs(self, obs, bdims, training=False):
    return self.sub(
        'obs_encoder', IssuePairObsEncoder,
        max_issues=self.max_issues, max_values=self.max_values,
        act=self.act, norm=self.norm, winit=self.winit,
        **self.obs_encoder_kw)(obs, bdims, training=training)

  def encode_action(self, act, obs, bdims):
    action_emb = self.sub(
        'action_encoder', ActionEncoder,
        max_issues=self.max_issues, max_values=self.max_values,
        action_units=self.action_units, act=self.act, norm=self.norm,
        winit=self.winit, **self.action_encoder_kw)(
            act['action_type'], act['offer_values'], obs['issue_mask'], bdims)
    return {'action_emb': action_emb}

  def observe(self, carry, obs, act, reset, training, single=False):
    bdims = 1 if single else 2
    tokens, obs_metrics = self.encode_obs(obs, bdims, training=training)
    dyn_act = self.encode_action(act, obs, bdims)
    dyn = self.sub('dyn', rssm.RSSM, self.dyn_act_space, **self.dyn_kw)
    if single:
      carry, entries, feat = dyn.observe(
          carry, tokens, dyn_act, reset, training, single=True)
    else:
      carry, entries, feat = dyn.observe(
          carry, tokens, dyn_act, reset, training)
    feat_tensor = feat_to_tensor(feat)
    pred, osi_metrics = self.sub(
        'osi', OpponentTableOSI,
        max_issues=self.max_issues, max_values=self.max_values,
        act=self.act, norm=self.norm, winit=self.winit,
        **self.osi_kw)(feat_tensor, obs, bdims)
    osi_loss, osi_loss_metrics = self.sub(
        'osi', OpponentTableOSI,
        max_issues=self.max_issues, max_values=self.max_values,
        act=self.act, norm=self.norm, winit=self.winit,
        **self.osi_kw).loss(pred, obs)
    metrics = {}
    metrics.update(obs_metrics)
    metrics.update(osi_metrics)
    metrics.update(osi_loss_metrics)
    return carry, entries, feat, pred, osi_loss, metrics


def feat_to_tensor(feat):
  return jnp.concatenate([
      nn.cast(feat['deter']),
      nn.cast(feat['stoch'].reshape((*feat['stoch'].shape[:-2], -1)))], -1)
