# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file implements common dense2sparse pruning algorithms."""
import dataclasses
import chex
import jax
import jax.numpy as jnp
from jaxpruner import base_updater
import flax
import functools
import optax
from jaxpruner import sparsity_types

BaseUpdater = base_updater.BaseUpdater

def estimate_neuron_score(params, activations_dict, prune_layers):
  def estimate_single_layer_neuron_scores(activation):
    reduce_axes = list(range(activation.ndim - 1))
    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    score /= jnp.mean(score) + 1e-9
    return score

  param_dict = flax.traverse_util.flatten_dict(params, sep='/')
  for i, layer_name in enumerate(prune_layers):
    param_key = 'params/' + layer_name + '/kernel'
    activation = activations_dict[layer_name + '_act/__call__'][0]
    score = estimate_single_layer_neuron_scores(activation)
    # print(score.shape, param_dict[param_key].shape) # (32,) (8, 8, 4, 32); (512,) (7744, 512)
    # So we only prune incoming weights of dormant neurons?
    # Yes or no, it's basically the same since after masking the incoming weights the outgoing weights become useless.
    param_dict[param_key] = score
  return params

@dataclasses.dataclass
class ActivationPruning(BaseUpdater):
  """Implements activation based pruning."""
  sparsity_type: sparsity_types.SparsityType = sparsity_types.Dormant()#Unstructured()#Dormant()

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    scores = estimate_neuron_score(params, self.activations_dict, self.prune_layers)
    return scores

  def pre_forward_update(
      self, activations_dict, prune_layers, opt_state
  ):
    """Used to transform paramaters before forward pass."""
    del opt_state
    self.activations_dict = activations_dict
    self.prune_layers = prune_layers

  def create_masks(self, scores, sparsities):
    def topk_ifnot_none(score, sparsity):
      self.topk_fn = functools.partial(self.topk_fn, )
      return None if sparsity is None else self.topk_fn(score, sparsity)
    return jax.tree_util.tree_map(topk_ifnot_none, scores, sparsities)
    # return self.topk_fn(scores, self.threshold)

  def wrap_optax(
      self, inner
  ):
    """Wraps an existing optax transformation and adds sparsity related updates.

    The gradient transformation provided (`inner`) is called as it is at
    initialization and during the update steps. In addition to this, a sparse
    state is created using the `self.init_state` function, which includes
    variables like masks needed by the algorithms. The sparse state is updated
    according to the given schedule using the `self.update_state` function.

    The returned transformation is a GradientTransformationExtraArgs: its
    update function accepts arbitrary additional args which it passes to
    `self.update_state`.

    Args:
      inner: An optax gradient transformation.

    Returns:
      An updated optax gradient transformation.
    """

    def init_fn(params):
      sparse_state = self.init_state(params)
      sparse_state = sparse_state._replace(inner_state=inner.init(params))
      return sparse_state

    def update_fn(updates, state, params, is_reset, **kwargs):
      is_update_step = self.scheduler.is_mask_update_iter(state.count)
      no_update_op = lambda state, *_: state
      if isinstance(is_reset, bool) and is_reset:
        new_state = jax.lax.cond(
            is_update_step,
            functools.partial(self.update_state, **kwargs),
            no_update_op,
            state,
            params,
            updates,
        )
      else:
        new_state = state
        
      if self.is_sparse_gradients:
        new_updates = self.apply_masks(updates, new_state.masks)
      else:
        new_updates = updates

      should_skip_inner = jnp.logical_and(is_update_step, self.skip_gradients)

      def no_inner_update(updates, inner_state, params):
        # Set gradients to zero and don't update the step.
        del params
        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
        return zero_updates, inner_state

      new_updates, new_inner_state = jax.lax.cond(
          should_skip_inner,
          no_inner_update,
          inner.update,
          new_updates,
          new_state.inner_state,
          params,
      )
      new_state = new_state._replace(
          count=optax.safe_int32_increment(new_state.count),
          inner_state=new_inner_state,
      )
      return new_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@dataclasses.dataclass
class MagnitudePruning(BaseUpdater):
  """Implements magnitude based pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state, grads
    param_magnitudes = jax.tree_util.tree_map(jnp.abs, params)
    return param_magnitudes


@dataclasses.dataclass
class SaliencyPruning(BaseUpdater):
  """Implements saliency (magnitude*gradient) based pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del sparse_state
    saliencies = jax.tree_util.tree_map(lambda p, g: jnp.abs(p * g), params, grads)
    return saliencies


def generate_random_scores(
    params, rng_seed
):
  """Generates random values matching the shape in params tree."""
  num_vars = len(jax.tree_util.tree_leaves(params))
  treedef = jax.tree_util.tree_structure(params)
  all_keys = jax.random.split(rng_seed, num=num_vars)

  return jax.tree_util.tree_map(
      lambda p, k: jax.random.uniform(k, shape=p.shape, dtype=p.dtype),
      params,
      jax.tree_util.tree_unflatten(treedef, all_keys),
  )


@dataclasses.dataclass
class RandomPruning(BaseUpdater):
  """Implements random pruning."""

  def calculate_scores(self, params, sparse_state=None, grads=None):
    del grads
    if sparse_state is None:
      new_rng = self.rng_seed
    else:
      new_rng = jax.random.fold_in(self.rng_seed, sparse_state.count)

    random_scores = generate_random_scores(params, new_rng)

    # Apply mask so that we ensure pruned connections have lowest score.
    if sparse_state is not None:
      random_scores = self.apply_masks(random_scores, sparse_state.masks)
    return random_scores
