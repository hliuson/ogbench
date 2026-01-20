import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')


def save_pretrain_state(state, target_params, save_dir, step, prefix='pretrain'):
    """Save a pretraining state and target parameters to a file."""
    save_dict = dict(
        state=flax.serialization.to_state_dict(state),
        target_params=flax.serialization.to_state_dict(target_params),
    )
    save_path = os.path.join(save_dir, f'{prefix}_checkpoint_{step}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_pretrain_state(state, target_params, restore_path, restore_step, prefix='pretrain'):
    """Restore a pretraining state and target parameters from a file."""
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/{prefix}_checkpoint_{restore_step}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    state = flax.serialization.from_state_dict(state, load_dict['state'])
    target_params = flax.serialization.from_state_dict(target_params, load_dict['target_params'])

    print(f'Restored from {restore_path}')
    return state, target_params


def save_encoder_params(encoder_params, save_dir, step, prefix='encoder'):
    """Save encoder parameters to a file."""
    save_dict = dict(
        encoder_params=flax.serialization.to_state_dict(encoder_params),
    )
    save_path = os.path.join(save_dir, f'{prefix}_params_{step}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def load_encoder_params(restore_path, restore_step, prefix='encoder'):
    """Load encoder parameters from a file."""
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/{prefix}_params_{restore_step}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    print(f'Loaded encoder params from {restore_path}')
    return load_dict['encoder_params']


def _tree_matches(tree, template):
    if isinstance(template, (dict, flax.core.FrozenDict)):
        if not isinstance(tree, (dict, flax.core.FrozenDict)):
            return False
        if set(tree.keys()) != set(template.keys()):
            return False
        return all(_tree_matches(tree[k], template[k]) for k in template.keys())
    return np.shape(tree) == np.shape(template)


def _path_matches_key(path, match_key):
    """Check if any path component contains the match_key as a substring.

    This allows match_key='high_encoder' to match paths like ('modules_high_encoder', 'encoder').
    """
    return any(match_key in component for component in path)


def inject_encoder_params(params, encoder_params, match_key=None):
    """Inject encoder parameters into any matching subtree of params.

    Args:
        params: The full parameter tree to inject into.
        encoder_params: The encoder parameters to inject.
        match_key: Optional string to filter which subtrees to inject into.
            Uses substring matching, so 'high_encoder' will match paths
            containing 'modules_high_encoder', 'high_encoder', etc.

    Returns:
        Tuple of (new_params, count, injected_paths) where count is the number
        of injections and injected_paths is a list of path tuples.
    """
    params_is_frozen = isinstance(params, flax.core.FrozenDict)
    params_dict = flax.core.unfreeze(params) if params_is_frozen else params
    encoder_params_dict = (
        flax.core.unfreeze(encoder_params)
        if isinstance(encoder_params, flax.core.FrozenDict)
        else encoder_params
    )
    count = 0
    injected_paths = []

    def replace_subtree(path, tree):
        nonlocal count
        if (match_key is None or _path_matches_key(path, match_key)) and _tree_matches(tree, encoder_params_dict):
            count += 1
            injected_paths.append(path)
            return encoder_params_dict
        if isinstance(tree, dict):
            return {k: replace_subtree(path + (k,), v) for k, v in tree.items()}
        return tree

    new_params = replace_subtree((), params_dict)
    if count == 0:
        if match_key is None:
            raise ValueError('No matching encoder subtrees found to inject.')
        raise ValueError(f'No matching encoder subtrees found to inject with match_key={match_key}.')
    if params_is_frozen:
        new_params = flax.core.freeze(new_params)
    return new_params, count, injected_paths


def print_param_tree(params, indent=0, max_depth=None):
    """Print the parameter tree structure with shapes.

    Args:
        params: The parameter tree to print.
        indent: Current indentation level.
        max_depth: Maximum depth to print (None for unlimited).
    """
    if max_depth is not None and indent >= max_depth:
        return

    prefix = '  ' * indent
    if isinstance(params, (dict, flax.core.FrozenDict)):
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, (dict, flax.core.FrozenDict)):
                print(f'{prefix}{key}/')
                print_param_tree(value, indent + 1, max_depth)
            else:
                shape = np.shape(value)
                print(f'{prefix}{key}: {shape}')
    else:
        shape = np.shape(params)
        print(f'{prefix}{shape}')
