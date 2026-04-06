import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if hasattr(self, 'valid_idxs'):
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - use_dual_value_goals: Whether to additionally provide an explicit pair of
      value goals:
        - one guaranteed in-trajectory goal (`*_intraj`)
        - one guaranteed random goal (`*_random`)
      This does not change the main mixed `value_goals` field.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )
        if self.config.get('use_dual_value_goals', False):
            assert self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] > 0, (
                'use_dual_value_goals requires nonzero in-trajectory value-goal mass'
            )

        if self.config.get('agent_name') in (
            'trl',
            'latent_trl',
            'ltrl_sharsa',
            'ltrl_hiql',
            'state_trl',
            'vae_trl',
        ):
            cur_idx = 0
            valid_idxs = []
            for terminal_idx in self.terminal_locs:
                valid_idxs.append(np.arange(cur_idx, terminal_idx))
                cur_idx = terminal_idx + 1
            valid_idxs = np.concatenate(valid_idxs)
            valids = np.zeros(self.size, dtype=np.float32)
            valids[valid_idxs] = 1.0
            self.dataset = Dataset(self.dataset.copy(dict(valids=valids)))

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        midpoint_agent_names = (
            'trl',
            'latent_trl',
            'ltrl_sharsa',
            'ltrl_hiql',
            'state_trl',
            'vae_trl',
        )
        need_intraj_mask = (
            self.config['value_p_randomgoal'] > 0
            and (
                (
                    self.config.get('agent_name') in midpoint_agent_names
                    and (
                        self.config.get('z_proposal_coef', 0) > 0
                        or self.config.get('use_dual_value_goals', False)
                    )
                )
                or self.config.get('need_value_intraj_mask', False)
            )
        )
        if need_intraj_mask:
            value_goal_idxs, value_is_intraj = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
                return_intraj_mask=True,
            )
        else:
            value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                self.config['value_geom_sample'],
            )
        actor_need_intraj_mask = (
            (
                self.config.get('agent_name') in midpoint_agent_names
                or self.config.get('need_actor_nstep', False)
            )
            and self.config.get('q_action_n_step', 1) > 1
        )
        if actor_need_intraj_mask:
            actor_goal_idxs, actor_is_intraj = self.sample_goals(
                idxs,
                self.config['actor_p_curgoal'],
                self.config['actor_p_trajgoal'],
                self.config['actor_p_randomgoal'],
                self.config['actor_geom_sample'],
                return_intraj_mask=True,
            )
        else:
            actor_goal_idxs = self.sample_goals(
                idxs,
                self.config['actor_p_curgoal'],
                self.config['actor_p_trajgoal'],
                self.config['actor_p_randomgoal'],
                self.config['actor_geom_sample'],
            )

        batch['value_goals'] = self.get_goal_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_goal_observations(actor_goal_idxs)
        if self.config.get('use_dual_value_goals', False):
            intraj_mass = self.config['value_p_curgoal'] + self.config['value_p_trajgoal']
            intraj_p_cur = self.config['value_p_curgoal'] / intraj_mass
            value_goal_idxs_intraj = self.sample_goals(
                idxs,
                intraj_p_cur,
                1.0 - intraj_p_cur,
                0.0,
                self.config['value_geom_sample'],
            )
            value_goal_idxs_random = self.sample_goals(
                idxs,
                0.0,
                0.0,
                1.0,
                self.config['value_geom_sample'],
            )
            batch['value_goals_intraj'] = self.get_goal_observations(value_goal_idxs_intraj)
            batch['value_goals_random'] = self.get_goal_observations(value_goal_idxs_random)
            # Legacy alias. No current agent consumes this, but keeping it avoids
            # breaking any older ad hoc analysis code.
            batch['value_goals_cf'] = batch['value_goals_random']
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config.get('agent_name') in midpoint_agent_names:
            final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
            assert (idxs != final_state_idxs).all()

            value_midpoint_idxs_intraj = None
            if need_intraj_mask:
                batch['value_goals_is_intraj'] = value_is_intraj.astype(np.float32)
                # For in-trajectory samples: midpoint between s and g on trajectory.
                # For counterfactual samples: midpoint is dummy (set to idxs; agent
                # routes these to the latent backup which ignores the state midpoint).
                intraj_goal_idxs = np.where(value_is_intraj, value_goal_idxs, final_state_idxs)
                intraj_goal_idxs = np.maximum(intraj_goal_idxs, idxs + 1)
                value_midpoint_idxs = np.random.randint(idxs, intraj_goal_idxs)
                value_midpoint_idxs = np.where(value_is_intraj, value_midpoint_idxs, idxs)
                if self.config.get('use_dual_value_goals', False):
                    intraj_midpoint_goal_idxs = np.maximum(value_goal_idxs_intraj, idxs + 1)
                    value_midpoint_idxs_intraj = np.random.randint(idxs, intraj_midpoint_goal_idxs)
            else:
                if self.config.get('use_dual_value_goals', False):
                    mixed_midpoint_goal_idxs = np.maximum(value_goal_idxs, idxs + 1)
                    intraj_midpoint_goal_idxs = np.maximum(value_goal_idxs_intraj, idxs + 1)
                    value_midpoint_idxs = np.random.randint(idxs, mixed_midpoint_goal_idxs)
                    value_midpoint_idxs_intraj = np.random.randint(idxs, intraj_midpoint_goal_idxs)
                else:
                    mixed_midpoint_goal_idxs = np.maximum(value_goal_idxs, idxs + 1)
                    value_midpoint_idxs = np.random.randint(idxs, mixed_midpoint_goal_idxs)

            batch['value_goal_observations'] = self.get_observations(value_goal_idxs)
            if self.config.get('agent_name') in {'ltrl_hiql', 'latent_trl', 'vae_trl'}:
                batch['actor_goal_observations'] = self.get_observations(actor_goal_idxs)
            else:
                batch['actor_goal_observations'] = self.get_observations(value_goal_idxs)
            if actor_need_intraj_mask:
                actor_n_step = int(self.config.get('q_action_n_step', 1))
                actor_nstep_idxs = np.minimum(idxs + actor_n_step, final_state_idxs)
                batch['actor_goals_is_intraj'] = actor_is_intraj.astype(np.float32)
                batch['actor_goal_offsets'] = (actor_goal_idxs - idxs) * actor_is_intraj
                batch['actor_nstep_observations'] = self.get_observations(actor_nstep_idxs)
                batch['actor_nstep_steps'] = actor_nstep_idxs - idxs
            if self.config.get('critic_n_step', 1) > 1:
                value_n_step = int(self.config.get('critic_n_step', 1))
                value_nstep_idxs = np.minimum(idxs + value_n_step, final_state_idxs)
                batch['value_nstep_observations'] = self.get_observations(value_nstep_idxs)
                batch['value_nstep_steps'] = value_nstep_idxs - idxs
            if self.config.get('use_dual_value_goals', False):
                batch['value_goal_observations_intraj'] = self.get_observations(value_goal_idxs_intraj)
                batch['value_goal_observations_random'] = self.get_observations(value_goal_idxs_random)
                batch['value_goal_observations_cf'] = batch['value_goal_observations_random']
                batch['value_offsets_intraj'] = value_goal_idxs_intraj - idxs
                batch['value_midpoint_offsets_intraj'] = value_midpoint_idxs_intraj - idxs
            if need_intraj_mask:
                # Only compute offsets for intraj; cf offsets are meaningless
                # (cross-trajectory) and would produce inf via discount^negative.
                batch['value_offsets'] = (value_goal_idxs - idxs) * value_is_intraj
            else:
                batch['value_offsets'] = value_goal_idxs - idxs
            batch['value_midpoint_offsets'] = value_midpoint_idxs - idxs
            batch['value_midpoint_observations'] = self.get_observations(value_midpoint_idxs)
            batch['value_midpoint_actions'] = self.dataset['actions'][value_midpoint_idxs]
            batch['next_actions'] = self.dataset['actions'][idxs + 1]
            if self.config.get('use_dual_value_goals', False):
                batch['value_midpoint_observations_intraj'] = self.get_observations(value_midpoint_idxs_intraj)
                batch['value_midpoint_actions_intraj'] = self.dataset['actions'][value_midpoint_idxs_intraj]

            if 'oracle_reps' in self.dataset:
                batch['value_midpoint_goals'] = self.dataset['oracle_reps'][value_midpoint_idxs]
                batch['value_cur_goals'] = self.dataset['oracle_reps'][idxs]
                batch['value_next_goals'] = self.dataset['oracle_reps'][idxs + 1]
                if self.config.get('use_dual_value_goals', False):
                    batch['value_midpoint_goals_intraj'] = self.dataset['oracle_reps'][value_midpoint_idxs_intraj]
            else:
                batch['value_midpoint_goals'] = self.get_observations(value_midpoint_idxs)
                batch['value_cur_goals'] = self.get_observations(idxs)
                batch['value_next_goals'] = self.get_observations(idxs + 1)
                if self.config.get('use_dual_value_goals', False):
                    batch['value_midpoint_goals_intraj'] = self.get_observations(value_midpoint_idxs_intraj)

            # Sample in-trajectory negatives for CRTR auxiliary loss.
            # Sample a random point from the same trajectory (different from midpoint).
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            traj_lengths = final_state_idxs - initial_state_idxs + 1
            # Sample random offset within trajectory, then shift to avoid midpoint
            intraj_neg_offsets = np.random.randint(0, traj_lengths - 1)
            intraj_neg_idxs = initial_state_idxs + intraj_neg_offsets
            # If we hit the midpoint, shift by 1
            intraj_neg_idxs = np.where(
                intraj_neg_idxs >= value_midpoint_idxs,
                intraj_neg_idxs + 1,
                intraj_neg_idxs,
            )
            # Clamp to trajectory bounds (in case we shifted past the end)
            intraj_neg_idxs = np.minimum(intraj_neg_idxs, final_state_idxs)
            batch['intraj_negative_observations'] = self.get_observations(intraj_neg_idxs)

            # Fixed-horizon target for hierarchical state predictor.
            # Sample s_{t+n} where n is fixed (clamped to trajectory bounds).
            hierarchical_horizon = self.config.get('hierarchical_horizon', 0)
            if hierarchical_horizon > 0:
                hier_target_idxs = np.minimum(idxs + hierarchical_horizon, final_state_idxs)
                batch['hierarchical_target_observations'] = self.get_observations(hier_target_idxs)
                batch['hierarchical_target_offsets'] = hier_target_idxs - idxs

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                aug_keys = ['observations', 'next_observations', 'value_goals', 'actor_goals']
                if self.config.get('agent_name') in midpoint_agent_names:
                    aug_keys.extend(
                        [
                            'value_goal_observations',
                            'actor_goal_observations',
                            'value_midpoint_observations',
                            'value_midpoint_goals',
                            'value_cur_goals',
                            'value_next_goals',
                            'intraj_negative_observations',
                        ]
                    )
                    if 'actor_nstep_observations' in batch:
                        aug_keys.append('actor_nstep_observations')
                    if self.config.get('use_dual_value_goals', False):
                        aug_keys.extend(
                            [
                                'value_goals_intraj',
                                'value_goals_random',
                                'value_goals_cf',
                                'value_goal_observations_intraj',
                                'value_goal_observations_random',
                                'value_goal_observations_cf',
                                'value_midpoint_observations_intraj',
                                'value_midpoint_goals_intraj',
                            ]
                        )
                    if 'hierarchical_target_observations' in batch:
                        aug_keys.append('hierarchical_target_observations')
                self.augment(batch, aug_keys)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None, return_intraj_mask=False):
        """Sample goals for the given indices.

        If return_intraj_mask is True, also returns a boolean mask indicating which
        samples used in-trajectory (trajgoal or curgoal) vs random (counterfactual) goals.
        """
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
            is_intraj = np.ones(batch_size, dtype=bool)
        else:
            is_traj = np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal)
            goal_idxs = np.where(is_traj, traj_goal_idxs, random_goal_idxs)

            # Goals at the current state.
            is_cur = np.random.rand(batch_size) < p_curgoal
            goal_idxs = np.where(is_cur, idxs, goal_idxs)
            is_intraj = is_traj | is_cur

        if return_intraj_mask:
            return goal_idxs, is_intraj
        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_goal_observations(self, idxs):
        """Return goal observations for the given indices.

        If oracle_reps exists in the dataset (oracle rep mode), returns those.
        Otherwise returns regular observations.
        """
        if 'oracle_reps' in self.dataset:
            return self.dataset['oracle_reps'][idxs]
        else:
            return self.get_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class ATCDataset:
    """Dataset class for ATC pretraining.

    This class samples anchor/positive observation pairs (o_t, o_{t+k}) from the same trajectory and supports
    frame stacking and random-shift augmentation for image observations.

    It reads the following keys from the config:
    - frame_stack: Number of frames to stack.
    - p_aug: Probability of applying image augmentation.
    - augment_padding: Padding size for random shift (default: 4).
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        self._atc_valid_cache = {}

        if self.config['frame_stack'] is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, k, evaluation=False):
        """Sample a batch of anchor/positive observations from the same trajectory."""
        valid_idxs = self.get_valid_atc_idxs(k)
        idxs = np.random.choice(valid_idxs, size=batch_size)

        batch = {
            'observations': self.get_observations(idxs),
            'positive_observations': self.get_observations(idxs + k),
        }

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'positive_observations'])

        return batch

    def get_valid_atc_idxs(self, k):
        """Return valid anchor indices for a given temporal offset k."""
        if k in self._atc_valid_cache:
            return self._atc_valid_cache[k]

        if 'valids' in self.dataset:
            candidate_idxs = self.dataset.valid_idxs
        else:
            candidate_idxs = np.arange(self.size)

        candidate_idxs = candidate_idxs[candidate_idxs + k < self.size]
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, candidate_idxs)]
        mask = candidate_idxs + k <= final_state_idxs
        valid_idxs = candidate_idxs[mask]

        if len(valid_idxs) == 0:
            raise ValueError(f'No valid ATC indices found for k={k}.')

        self._atc_valid_cache[k] = valid_idxs
        return valid_idxs

    def augment(self, batch, keys):
        """Apply random-shift augmentation to image observations."""
        padding = self.config.get('augment_padding', 4)
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support hierarchical goal-conditioned RL. It reads the following additional key from
    the config:
    - subgoal_steps (optional: value_subgoal_steps, actor_subgoal_steps): Subgoal steps. It is also possible to specify
        `value_subgoal_steps` and `actor_subgoal_steps` separately.
    - low_discount: If specified, return low-level value goals as well.
    """

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, subgoal_steps):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # Clip to the end of the trajectory.
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def get_high_actions(self, target_idxs, cur_idxs):
        return self.get_goal_observations(target_idxs)

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        high_subgoal_steps = self.config.get('high_subgoal_steps', self.config['subgoal_steps'])
        value_subgoal_steps = (
            high_subgoal_steps if self.config.get('value_subgoal_steps') is None else self.config['value_subgoal_steps']
        )
        high_value_next_idxs, high_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_subgoal_steps,
        )

        batch['high_value_reps'] = batch['observations']
        batch['high_value_goals'] = self.get_goal_observations(high_value_goal_idxs)
        batch['high_value_goal_observations'] = self.get_observations(high_value_goal_idxs)
        batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
        # high_value_next_observations is the next state observation, not a goal
        batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        batch['high_value_offsets'] = high_value_goal_idxs - idxs

        high_value_successes = (high_value_subgoal_steps < value_subgoal_steps).astype(float)
        batch['high_value_subgoal_steps'] = high_value_subgoal_steps
        batch['high_value_masks'] = 1.0 - high_value_successes
        if self.config['gc_negative']:
            batch['high_value_rewards'] = -(1 - self.config['discount'] ** high_value_subgoal_steps) / (
                1 - self.config['discount']
            )
        else:
            batch['high_value_rewards'] = (self.config['discount'] ** high_value_subgoal_steps) * high_value_successes

        low_subgoal_steps = self.config.get('low_subgoal_steps', self.config['subgoal_steps'])
        low_value_next_idxs, low_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            low_subgoal_steps,
        )
        batch['low_value_next_observations'] = self.get_observations(low_value_next_idxs)

        low_value_successes = (low_value_subgoal_steps < low_subgoal_steps).astype(float)
        batch['low_value_subgoal_steps'] = low_value_subgoal_steps
        batch['low_value_masks'] = 1.0 - low_value_successes
        if self.config['gc_negative']:
            batch['low_value_rewards'] = -(1 - self.config['discount'] ** low_value_subgoal_steps) / (
                1 - self.config['discount']
            )
        else:
            batch['low_value_rewards'] = (self.config['discount'] ** low_value_subgoal_steps) * low_value_successes

        # Sample low-level value goals (if requested).
        if self.config.get('low_discount') is not None:
            low_value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                geom_sample=True,
                discount=self.config['low_discount'],
            )

            batch['low_value_goals'] = self.get_goal_observations(low_value_goal_idxs)
            successes = (idxs == low_value_goal_idxs).astype(float)
            batch['low_value_masks'] = 1.0 - successes
            batch['low_value_rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # One-step information (for compatibility with HIQL and other agents).
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['value_goals'] = batch['high_value_goals']
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample high-level actor goals.
        high_actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )
        actor_subgoal_steps = (
            high_subgoal_steps if self.config.get('actor_subgoal_steps') is None else self.config['actor_subgoal_steps']
        )
        high_actor_next_idxs, high_actor_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            actor_subgoal_steps,
        )

        batch['high_actor_goals'] = self.get_goal_observations(high_actor_goal_idxs)
        batch['high_actor_goal_observations'] = self.get_observations(high_actor_goal_idxs)
        batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
        # high_actor_next_observations is the next state observation, not a goal
        batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)
        # For HIQL compatibility: high_actor_targets = high_actor_actions
        batch['high_actor_targets'] = batch['high_actor_actions']

        # Compute low-level actor goals.
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)
        batch['low_actor_goals'] = self.get_high_actions(low_actor_goal_idxs, idxs)
        batch['low_actor_goal_observations'] = self.get_observations(low_actor_goal_idxs)
        low_actor_next_idxs, _ = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            low_subgoal_steps,
        )
        batch['low_actor_next_observations'] = self.get_observations(low_actor_next_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'high_value_goals',
                        'high_value_goal_observations',
                        'high_value_actions',
                        'high_value_next_observations',
                        'low_value_next_observations',
                        'low_actor_goals',
                        'low_actor_goal_observations',
                        'low_actor_next_observations',
                        'high_actor_goals',
                        'high_actor_goal_observations',
                        'high_actor_actions',
                        'high_actor_next_observations',
                        'high_actor_targets',
                    ],
                )

        return batch
