"""LeRobot v3 adapter for the dual-arm Piper-X dataset used in this workspace."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    apply_normalization,
    materialize_eef_stats,
)
from openwam.dataloader.utils.piperx_kinematics import EEF20_DIM, JOINT14_DIM, joint14_to_eef20

logger = logging.getLogger(__name__)

ACTION_MODE = "eef"
ACTION_STATS_KEY = ACTION_MODE
STATE_STATS_KEY = f"{ACTION_MODE}_state"
NORMALIZATION_STATS_FILENAME = "piperx_eef_normalization_stats.npy"
GRIPPER_CONVENTION = "minus1_closed_plus1_open_after_minmax"
REPRESENTATION = "absolute_dual_piper_x_flange_eef20"

EXPECTED_JOINT_NAMES = [
    *(f"left_joint_{index}.pos" for index in range(1, 7)),
    "left_gripper.pos",
    *(f"right_joint_{index}.pos" for index in range(1, 7)),
    "right_gripper.pos",
]


class PiperXLeRobotDataset(LeRobotV3Reader):
    """Convert 14-D dual-Piper-X joint targets and state to absolute EEF20."""

    DATASET_NAME = "DualPiperXLeRobot"
    ACTION_DIM = EEF20_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    PROMPT_FILE_REQUIRED = True
    DEFAULT_NORMALIZE_MODE = "min-max"
    DEPLOY_ACTION_MODE = ACTION_MODE
    DEFAULT_CAMERA_LAYOUT: ClassVar[tuple[str, str, str]] = (
        "observation.images.fixed_front",
        "observation.images.left_arm",
        "observation.images.right_arm",
    )
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "normalization_stats_path",
        "task_indices",
        "excluded_episode_indices",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = ACTION_MODE,
        normalization_stats_path: Optional[str] = None,
        task_indices: Optional[list[int]] = None,
        excluded_episode_indices: Optional[list[int]] = None,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != ACTION_MODE:
            raise ValueError(f"Dual Piper-X supports only action_mode={ACTION_MODE!r}, got {action_mode!r}")
        if unify_action and unify_action_map is None:
            raise ValueError('Dual Piper-X unify_action=true requires unify_action_map=["0-9", "34-43"]')
        self._task_indices = sorted(set(int(i) for i in task_indices)) if task_indices is not None else None
        self._excluded_episode_indices = sorted(set(int(i) for i in (excluded_episode_indices or [])))
        if self._task_indices == []:
            raise ValueError("task_indices must select at least one task")
        if (self._task_indices is not None or self._excluded_episode_indices) and not normalization_stats_path:
            raise ValueError("Filtered Piper-X data requires a separate normalization_stats_path")
        self.action_mode = mode
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=bool(unify_action),
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        if not self._multiview and self._target_camera is not None:
            return self._target_camera, None, None
        layout = list(self._camera_layout_param or self.DEFAULT_CAMERA_LAYOUT)
        layout += [None] * (3 - len(layout))
        return tuple(str(camera) if camera else None for camera in layout[:3])

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        # Base reader has already computed shard/video offsets. Preserve them
        # while selecting tasks; success labels are not used for this teleop data.
        selected = eps_df
        if self._task_indices is not None:
            tasks = pd.read_parquet(self._dataset_dir / "meta" / "tasks.parquet")
            names = dict(zip(tasks["task_index"].astype(int), tasks.index.astype(str)))
            missing = set(self._task_indices) - names.keys()
            if missing:
                raise ValueError(f"Unknown task_indices: {sorted(missing)}")
            wanted = {names[i] for i in self._task_indices}
            selected = selected[selected["tasks"].map(lambda values: any(t in wanted for t in values))]
        selected = selected[~selected["episode_index"].isin(self._excluded_episode_indices)]
        if selected.empty:
            raise ValueError("Piper-X task/episode selection contains no episodes")
        return selected.reset_index(drop=True)

    def _stats_selection(self) -> dict:
        return {
            "task_indices": self._task_indices,
            "excluded_episode_indices": self._excluded_episode_indices,
            "episode_indices": sorted(int(i) for i in self._eps_df["episode_index"]),
        }

    def _post_init(self, info: dict) -> None:
        if info.get("robot_type") != "bi_piper_follower":
            raise ValueError(
                f"Dual Piper-X reader requires robot_type='bi_piper_follower', got {info.get('robot_type')!r}"
            )
        features = info.get("features", {}) or {}
        for column in ("action", "observation.state"):
            feature = features.get(column, {})
            shape = tuple(feature.get("shape", ()))
            if shape != (JOINT14_DIM,):
                raise ValueError(f"Dual Piper-X {column} must have shape [{JOINT14_DIM}], got {shape}")
            names = feature.get("names")
            if names is not None and list(names) != EXPECTED_JOINT_NAMES:
                raise ValueError(f"Dual Piper-X {column} joint order does not match the required 14-D layout")
        missing_cameras = [camera for camera in self.DEFAULT_CAMERA_LAYOUT if camera not in features]
        if missing_cameras:
            raise ValueError(f"Dual Piper-X dataset is missing video features: {missing_cameras}")
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            raise ValueError(
                f"Dual Piper-X requires height={expected_size[0]}, width={expected_size[1]} in "
                f"{'multiview' if self._multiview else 'single-view'} mode"
            )

    def _build_stats_rank0(self, path: Path) -> None:
        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        rank = dist.get_rank() if dist_ready else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.piperx_lerobot_stats_computation import (
                build_and_save_piper_stats,
            )

            logger.info("No Piper EEF stats at %s; scanning the dataset on rank 0", path)
            build_and_save_piper_stats(
                self._dataset_dir,
                output=path,
                **self._stats_selection(),
            )
            return
        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build Piper stats: {path}")
            time.sleep(interval)

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            self._state_normalization_stats = None
            return None
        stats_path = (
            Path(self._source_stats_path)
            if self._source_stats_path
            else self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
        )
        if not stats_path.is_file():
            self._build_stats_rank0(stats_path)
        raw = np.load(stats_path, allow_pickle=True).item()
        if not isinstance(raw, dict) or ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
            raise KeyError(f"{stats_path} must contain {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")
        if getattr(self, "_task_indices", None) is not None or getattr(self, "_excluded_episode_indices", []):
            if raw.get("selection") != self._stats_selection():
                raise ValueError(f"{stats_path} was computed for a different episode/task selection; regenerate it")
        for name in (ACTION_STATS_KEY, STATE_STATS_KEY):
            block = raw[name]
            if block.get("representation") != REPRESENTATION:
                raise ValueError(f"{stats_path}:{name} has an incompatible representation")
            if block.get("gripper_convention") != GRIPPER_CONVENTION:
                raise ValueError(f"{stats_path}:{name} has an incompatible gripper convention")
        action_stats = materialize_eef_stats(
            dict(raw[ACTION_STATS_KEY]),
            self._normalize_mode,
            dim=EEF20_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}:{ACTION_STATS_KEY}",
            force_rot6d_identity=True,
        )
        self._state_normalization_stats = materialize_eef_stats(
            dict(raw[STATE_STATS_KEY]),
            self._normalize_mode,
            dim=EEF20_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}:{STATE_STATS_KEY}",
            force_rot6d_identity=True,
        )
        # The source file is already the deployment artifact. Keep both the
        # action block used for output unnormalization and the state block used
        # for proprio normalization when the trainer copies it into the run.
        self.normalization_stats_path = str(stats_path)
        return action_stats

    @staticmethod
    def _read_joint14(win: pd.DataFrame, column: str) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != JOINT14_DIM:
            raise ValueError(f"Dual Piper-X {column} must be (T, {JOINT14_DIM}), got {values.shape}")
        return values

    def _normalize_eef(self, values: np.ndarray, stats: Optional[dict]) -> np.ndarray:
        raw = joint14_to_eef20(values)
        normalized = np.array(apply_normalization(raw, stats, self._normalize_mode), dtype=np.float32, copy=True)
        normalized[..., ROT6D_DIMS_EEF20] = raw[..., ROT6D_DIMS_EEF20]
        return normalized

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        return self._normalize_eef(self._read_joint14(win, "action"), self._normalization_stats)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        state = self._read_joint14(win, "observation.state")[:1]
        return self._normalize_eef(state, self._state_normalization_stats)


__all__ = [
    "ACTION_MODE",
    "ACTION_STATS_KEY",
    "GRIPPER_CONVENTION",
    "NORMALIZATION_STATS_FILENAME",
    "PiperXLeRobotDataset",
    "REPRESENTATION",
    "STATE_STATS_KEY",
]
