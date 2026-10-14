"""Compute EEF20 normalization statistics for a dual-Piper-X LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.piperx_lerobot import (
    ACTION_SOURCE_COLUMNS,
    ACTION_STATS_KEY,
    GRIPPER_CONVENTION,
    NORMALIZATION_STATS_FILENAME,
    REPRESENTATION,
    STATE_STATS_KEY,
    validate_action_source,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.piperx_kinematics import joint14_to_eef20

STATE_COLUMN = "observation.state"


def _feature_stats(values: np.ndarray, *, semantics: str) -> dict:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    stats = {
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "q01": quantiles[0].astype(np.float64).tolist(),
        "q10": quantiles[1].astype(np.float64).tolist(),
        "q50": quantiles[2].astype(np.float64).tolist(),
        "q90": quantiles[3].astype(np.float64).tolist(),
        "q99": quantiles[4].astype(np.float64).tolist(),
        "count": [int(values.shape[0])],
        "num_timesteps": int(values.shape[0]),
        "representation": REPRESENTATION,
        "gripper_convention": GRIPPER_CONVENTION,
        "semantics": semantics,
    }
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    return stats


def compute_piper_stats(
    dataset_dir: str | Path,
    *,
    action_source: str = "target_action",
    task_indices: list[int] | None = None,
    excluded_episode_indices: list[int] | None = None,
    episode_indices: list[int] | None = None,
) -> dict:
    """Fit separate action/state EEF stats on the selected episodes only."""
    action_source = validate_action_source(action_source)
    action_column = ACTION_SOURCE_COLUMNS[action_source]
    root = Path(dataset_dir)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    actions, states, selected_episodes = [], [], set()
    for path in paths:
        # Reading the whole flat schema also handles literal dotted column names.
        frame = pq.read_table(path, memory_map=True).to_pandas()
        if task_indices is not None:
            frame = frame[frame["task_index"].isin(task_indices)]
        if episode_indices is not None:
            frame = frame[frame["episode_index"].isin(episode_indices)]
        frame = frame[~frame["episode_index"].isin(excluded_episode_indices or [])]
        if frame.empty:
            continue
        selected_episodes.update(int(i) for i in frame["episode_index"].unique())
        state_eef = joint14_to_eef20(np.stack(frame[STATE_COLUMN].values))
        states.append(state_eef)
        actions.append(
            state_eef if action_source == "future_state" else joint14_to_eef20(np.stack(frame[action_column].values))
        )
    if not actions:
        raise ValueError("Piper-X statistics selection contains no frames")
    if episode_indices is not None and selected_episodes != set(episode_indices):
        raise ValueError("Piper-X statistics could not find all selected episodes")
    action_semantics = (
        "absolute commanded flange pose"
        if action_source == "target_action"
        else "absolute achieved next-state flange pose"
    )
    return {
        "action_source": action_source,
        ACTION_STATS_KEY: _feature_stats(np.concatenate(actions), semantics=action_semantics),
        STATE_STATS_KEY: _feature_stats(np.concatenate(states), semantics="absolute measured flange pose"),
        "selection": {
            "task_indices": sorted(set(task_indices)) if task_indices is not None else None,
            "excluded_episode_indices": sorted(set(excluded_episode_indices or [])),
            "episode_indices": sorted(selected_episodes),
        },
    }


def build_and_save_piper_stats(
    dataset_dir: str | Path,
    output: str | Path | None = None,
    **selection,
) -> Path:
    """Compute and atomically save the stats payload."""
    root = Path(dataset_dir)
    out = Path(output) if output else root / "meta" / NORMALIZATION_STATS_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_piper_stats(root, **selection)
    fd, tmp_name = tempfile.mkstemp(dir=str(out.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--action-source", choices=list(ACTION_SOURCE_COLUMNS), default="target_action")
    parser.add_argument("--task-indices", type=int, nargs="+", default=None)
    parser.add_argument("--exclude-episodes", type=int, nargs="*", default=[])
    args = parser.parse_args()
    output = build_and_save_piper_stats(
        args.dataset_dir,
        args.output,
        action_source=args.action_source,
        task_indices=args.task_indices,
        excluded_episode_indices=args.exclude_episodes,
    )
    payload = np.load(output, allow_pickle=True).item()
    print(f"wrote {output}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: {block['num_timesteps']} rows, xyz mean={np.round(block['mean'][:3], 5).tolist()}")


if __name__ == "__main__":
    main()
