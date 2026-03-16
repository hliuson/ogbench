"""Chop a navigate dataset into short segments to create a stitch-like dataset.

Navigate datasets have long trajectories where the agent repeatedly reaches
random goals. Chopping them into short segments creates data that requires
stitching to solve long-horizon tasks.

Usage:
    # Single file:
    python chop_navigate_to_stitch.py \
        --input_path=/path/to/navigate-v0.npz \
        --output_path=/path/to/stitch-v0.npz \
        --segment_length=200

    # Directory of shards:
    python chop_navigate_to_stitch.py \
        --input_path=/path/to/navigate-100m-v0/ \
        --output_path=/path/to/stitch-100m-v0/ \
        --segment_length=200
"""

import glob
import os
import pathlib

import numpy as np
from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'Path to input navigate dataset (file or directory of shards).')
flags.DEFINE_string('output_path', None, 'Path to output stitch dataset (file or directory).')
flags.DEFINE_integer('segment_length', 200, 'Length of each segment.')


def chop_file(input_path, output_path):
    """Chop a single navigate .npz file into short segments."""
    print(f'  Loading {input_path}...')
    f = np.load(input_path)

    keys = list(f.keys())
    n = len(f['terminals'])

    terminals = f['terminals'].astype(bool)
    traj_ends = np.where(terminals)[0]

    # Build new terminals: mark every segment_length steps within each trajectory.
    # Since all trajectories have the same length, compute segment boundaries vectorized.
    seg_len = FLAGS.segment_length
    traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])
    traj_lengths = traj_ends - traj_starts + 1

    new_terminals = np.zeros(n, dtype=bool)
    new_terminals[traj_ends] = True  # Keep original boundaries.

    # For each trajectory, insert boundaries every seg_len steps.
    for traj_start, traj_len in zip(traj_starts, traj_lengths):
        # Indices within this trajectory where we insert boundaries (excluding the last step which is already terminal).
        offsets = np.arange(seg_len, traj_len, seg_len)
        if len(offsets) > 0:
            # The boundary is at the end of each segment (offset - 1 relative to traj_start).
            new_terminals[traj_start + offsets - 1] = True

    new_traj_ends = np.where(new_terminals)[0]
    seg_lengths = np.diff(np.concatenate([[-1], new_traj_ends]))
    print(f'  {len(traj_ends)} trajs -> {len(new_traj_ends)} segments '
          f'(mean={seg_lengths.mean():.1f}, min={seg_lengths.min()}, max={seg_lengths.max()})')

    dataset = {}
    for k in keys:
        data = f[k][...]
        if k == 'terminals':
            data = new_terminals
        dataset[k] = data

    print(f'  Saving {output_path}...')
    np.savez_compressed(output_path, **dataset)
    print(f'  Done.')
    return n, len(new_traj_ends)


def main(_):
    input_path = FLAGS.input_path
    output_path = FLAGS.output_path

    if os.path.isdir(input_path):
        # Directory of shards.
        shards = sorted(glob.glob(os.path.join(input_path, '*.npz')))
        print(f'Found {len(shards)} shard files in {input_path}')

        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        total_steps = 0
        total_segs = 0
        for shard in tqdm(shards, desc='Chopping shards'):
            fname = os.path.basename(shard)
            out_file = os.path.join(output_path, fname)
            n, nsegs = chop_file(shard, out_file)
            total_steps += n
            total_segs += nsegs

        print(f'\nTotal: {total_steps} steps, {total_segs} segments across {len(shards)} shards')
    else:
        # Single file.
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        chop_file(input_path, output_path)

    print('All done.')


if __name__ == '__main__':
    app.run(main)
