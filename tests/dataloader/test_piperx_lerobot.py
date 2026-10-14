from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from openwam.dataloader.piperx_lerobot import REPRESENTATION, PiperXLeRobotDataset
from openwam.dataloader.transforms.piperx_fk import PiperXForwardKinematics
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20
from openwam.dataloader.utils.piperx_kinematics import joint14_to_eef20
from openwam.dataloader.utils.stats_computation.piperx_lerobot_stats_computation import compute_piper_stats


def test_batched_fk_matches_supplied_scalar_pose():
    fk = PiperXForwardKinematics()
    joints = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            [0.5, 1.57, -1.57, 0.3, 0.2, 0.1],
        ]
    )
    matrices = fk.compute_fk_matrix(joints)
    for joint, matrix in zip(joints, matrices):
        pose = fk.compute_fk(joint)
        np.testing.assert_allclose(matrix[:3, 3], pose[:3], atol=1e-12)
        np.testing.assert_allclose(matrix[:3, :3], Rotation.from_euler("xyz", pose[3:]).as_matrix(), atol=1e-12)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)


def test_dual_arm_conversion_uses_piperx_flange_and_preserves_grips():
    joints = np.array([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.3, 0.2, -0.1, 0.4, -0.5, 0.6, 1.2]])
    converted = joint14_to_eef20(joints)
    fk = PiperXForwardKinematics()
    for joint_start, eef_start in [(0, 0), (7, 10)]:
        matrix = fk.compute_fk_matrix(joints[0, joint_start : joint_start + 6])
        np.testing.assert_allclose(converted[0, eef_start : eef_start + 3], matrix[:3, 3], atol=1e-7)
        np.testing.assert_allclose(converted[0, eef_start + 3 : eef_start + 6], matrix[:3, 0], atol=1e-7)
        np.testing.assert_allclose(converted[0, eef_start + 6 : eef_start + 9], matrix[:3, 1], atol=1e-7)
        assert converted[0, eef_start + 9] == np.float32(joints[0, joint_start + 6])


@pytest.mark.parametrize("values", [np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)])
def test_joint_conversion_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        joint14_to_eef20(values)


def test_filter_keeps_successful_teleop_despite_stale_labels_and_preserves_offsets(tmp_path: Path):
    (tmp_path / "meta").mkdir()
    pd.DataFrame({"task_index": [0, 1, 4]}, index=["pens", "towel", "bowls"]).to_parquet(
        tmp_path / "meta/tasks.parquet"
    )
    reader = object.__new__(PiperXLeRobotDataset)
    reader._dataset_dir = tmp_path
    reader._task_indices = [0, 4]
    reader._excluded_episode_indices = [24]
    eps = pd.DataFrame(
        {
            "episode_index": [0, 24, 25, 50, 200],
            "tasks": [["pens"], ["pens"], ["pens"], ["towel"], ["bowls"]],
            "episode_success": ["failure"] * 5,
            "_data_row_offset": [0, 100, 147, 200, 250],
            "_video_frame_offset/cam": [0, 100, 147, 200, 250],
        }
    )
    filtered = reader._filter_episodes(eps)
    assert filtered.episode_index.tolist() == [0, 25, 200]
    assert filtered["_data_row_offset"].tolist() == [0, 147, 250]
    assert filtered["_video_frame_offset/cam"].tolist() == [0, 147, 250]


def test_stats_exclude_other_tasks_and_deleted_episode(tmp_path: Path):
    (tmp_path / "data").mkdir()
    joints = np.zeros((4, 14))
    joints[:, [6, 13]] = np.array([[0.1, 0.2], [100, 100], [200, 200], [0.8, 0.9]])
    table = pa.table(
        {
            "action": joints.tolist(),
            "observation.state": joints.tolist(),
            "episode_index": [0, 24, 50, 200],
            "task_index": [0, 0, 1, 4],
        }
    )
    pq.write_table(table, tmp_path / "data/file-000.parquet")
    stats = compute_piper_stats(tmp_path, task_indices=[0, 4], excluded_episode_indices=[24])
    assert stats["selection"]["episode_indices"] == [0, 200]
    for key in ["eef", "eef_state"]:
        assert stats[key]["num_timesteps"] == 2
        assert stats[key]["representation"] == REPRESENTATION
        np.testing.assert_allclose(np.array(stats[key]["max"])[[9, 19]], [0.8, 0.9])
        np.testing.assert_array_equal(np.array(stats[key]["min"])[list(ROT6D_DIMS_EEF20)], -1)
        np.testing.assert_array_equal(np.array(stats[key]["max"])[list(ROT6D_DIMS_EEF20)], 1)


def test_rot6d_is_not_normalized():
    reader = object.__new__(PiperXLeRobotDataset)
    reader._normalize_mode = "min-max"
    joints = np.full((2, 14), 0.2)
    stats = {"min": np.full(20, -2.0), "max": np.full(20, 2.0)}
    raw = joint14_to_eef20(joints)
    normalized = reader._normalize_eef(joints, stats)
    np.testing.assert_array_equal(normalized[:, ROT6D_DIMS_EEF20], raw[:, ROT6D_DIMS_EEF20])


def test_multiview_keeps_both_arm_cameras_when_target_camera_is_set():
    reader = object.__new__(PiperXLeRobotDataset)
    reader._multiview = True
    reader._target_camera = "observation.images.fixed_front"
    reader._camera_layout_param = None
    assert reader._resolve_cameras({}) == reader.DEFAULT_CAMERA_LAYOUT


def test_wrong_selection_stats_are_rejected(tmp_path: Path):
    reader = object.__new__(PiperXLeRobotDataset)
    reader._normalize_mode = "min-max"
    reader._dataset_dir = tmp_path
    reader._task_indices = [0, 4]
    reader._excluded_episode_indices = [24]
    reader._eps_df = pd.DataFrame({"episode_index": [0, 200]})
    path = tmp_path / "wrong.npy"
    reader._source_stats_path = str(path)
    reader.action_source = "target_action"
    np.save(path, {"eef": {}, "eef_state": {}, "selection": {"episode_indices": [0, 24, 200]}})
    with pytest.raises(ValueError, match="different episode/task selection"):
        reader._load_stats({})


def _source_reader(source: str) -> PiperXLeRobotDataset:
    reader = object.__new__(PiperXLeRobotDataset)
    reader.action_source = source
    reader._normalize_mode = None
    reader._normalization_stats = None
    reader._state_normalization_stats = None
    reader._num_frames = 33
    reader._raw_action_dim = 20
    reader._unify = True
    reader._unify_dim = 80
    reader.ACTION_DIM = 80
    reader._unify_dst_index = np.array([*range(10), *range(34, 44)])
    reader._unify_state_dst_index = reader._unify_dst_index
    reader._unify_dim_mask = np.zeros(80, dtype=bool)
    reader._unify_dim_mask[reader._unify_dst_index] = True
    reader._unify_proprio_dim_mask = reader._unify_dim_mask.copy()
    return reader


@pytest.mark.parametrize("source", ["target_action", "future_state"])
@pytest.mark.parametrize("length", [33, 4, 2, 1])
def test_action_source_frame_alignment_and_boundary_mask(source, length):
    # Commands and achieved states intentionally have different joint and
    # gripper trajectories, so a mixed-source or unshifted label cannot pass.
    commands = np.full((length, 14), 0.1, dtype=np.float32)
    states = np.full((length, 14), -0.2, dtype=np.float32)
    commands[:, 0] += np.arange(length) * 0.01
    states[:, 0] += np.arange(length) * 0.02
    commands[:, 6] = 0.8 + np.arange(length) * 0.001
    commands[:, 13] = 0.9 - np.arange(length) * 0.001
    states[:, 6] = 0.2 + np.arange(length) * 0.005
    states[:, 13] = 0.3 + np.arange(length) * 0.006
    win = pd.DataFrame({"action": list(commands), "observation.state": list(states)})
    reader = _source_reader(source)
    action, mask = reader._finalize_action(reader._action_20d(win), length)
    proprio, proprio_mask = reader._finalize_proprio(reader._proprio_20d(win))
    expected_joints = states[1:] if source == "future_state" else commands
    expected = joint14_to_eef20(expected_joints)
    valid_steps = min(len(expected), 32)
    np.testing.assert_allclose(action[:valid_steps, reader._unify_dst_index], expected[:valid_steps], atol=1e-7)
    # Explicit gripper checks: both use the same source and +1 timing as the pose.
    np.testing.assert_allclose(action[:valid_steps, [9, 43]], expected_joints[:valid_steps, [6, 13]], atol=1e-7)
    assert mask[:valid_steps].sum() == valid_steps * 20
    assert not mask[valid_steps:].any()
    assert not mask[:, ~reader._unify_dim_mask].any()
    np.testing.assert_allclose(proprio[:, reader._unify_dst_index], joint14_to_eef20(states[:1]), atol=1e-7)
    assert proprio_mask.sum() == 20
    assert reader._train_min_window_len() == (2 if source == "future_state" else 1)


@pytest.mark.parametrize("source,expected_max", [("target_action", [0.8, 0.9]), ("future_state", [0.6, 0.7])])
def test_stats_follow_entire_selected_action_source(tmp_path: Path, source, expected_max):
    (tmp_path / "data").mkdir()
    commands = np.full((4, 14), 0.1)
    states = np.full((4, 14), -0.2)
    commands[:, [6, 13]] = [[0.1, 0.2], [100, 100], [200, 200], [0.8, 0.9]]
    states[:, [6, 13]] = [[0.3, 0.4], [300, 300], [400, 400], [0.6, 0.7]]
    pq.write_table(
        pa.table(
            {
                "action": commands.tolist(),
                "observation.state": states.tolist(),
                "episode_index": [0, 24, 50, 200],
                "task_index": [0, 0, 1, 4],
            }
        ),
        tmp_path / "data/file-000.parquet",
    )
    stats = compute_piper_stats(tmp_path, action_source=source, task_indices=[0, 4], excluded_episode_indices=[24])
    assert stats["action_source"] == source
    assert stats["selection"]["episode_indices"] == [0, 200]
    np.testing.assert_allclose(np.array(stats["eef"]["max"])[[9, 19]], expected_max)
    np.testing.assert_allclose(np.array(stats["eef_state"]["max"])[[9, 19]], [0.6, 0.7])
    if source == "future_state":
        for key in ["min", "max", "mean", "std", "q01", "q99"]:
            np.testing.assert_array_equal(stats["eef"][key], stats["eef_state"][key])
    else:
        assert stats["eef"]["mean"] != stats["eef_state"]["mean"]


def test_action_source_rejects_stale_stats(tmp_path: Path):
    reader = _source_reader("future_state")
    reader._normalize_mode = "min-max"
    path = tmp_path / "target_stats.npy"
    # Pre-existing stats without action_source have the target-action meaning.
    np.save(path, {"eef": {}, "eef_state": {}})
    reader._source_stats_path = str(path)
    with pytest.raises(ValueError, match="different action_source"):
        reader._load_stats({})


def test_legacy_target_stats_remain_usable(tmp_path: Path):
    reader = _source_reader("target_action")
    reader._normalize_mode = "min-max"
    path = tmp_path / "legacy.npy"
    block = {
        "min": [-1.0] * 20,
        "max": [1.0] * 20,
        "mean": [0.0] * 20,
        "std": [1.0] * 20,
        "representation": REPRESENTATION,
        "gripper_convention": "minus1_closed_plus1_open_after_minmax",
    }
    np.save(path, {"eef": block, "eef_state": block})
    reader._source_stats_path = str(path)
    assert reader._load_stats({}) is not None


def test_invalid_action_source_fails_before_dataset_loading(tmp_path: Path):
    with pytest.raises(ValueError, match="action_source"):
        PiperXLeRobotDataset(str(tmp_path), action_source="mixed_gripper")
