"""OpenPI chunk-serving protocol for the dual Piper-X EEF20 checkpoints."""

import asyncio
import logging
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.utils.piperx_kinematics import joint14_to_eef20
from openwam.deploy import msgpack_numpy
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.server import MAX_MESSAGE_BYTES

logger = logging.getLogger(__name__)

VIEWS = ("fixed_front", "left_arm", "right_arm")
SLOTS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
JOINT_NAMES = [
    f"{side}_{name}.pos" for side in ("left", "right") for name in [*(f"joint_{i}" for i in range(1, 7)), "gripper"]
]


class OpenPIPolicyServer:
    """Generate a fresh full chunk per request; no server-side action queue.

    Metadata is sent immediately on connection, followed by binary MessagePack
    request/response pairs. Inference errors are text frames, as in OpenPI.
    The engine is serialized across connections because its caches are shared.
    """

    def __init__(self, engine, cfg):
        if OmegaConf.select(cfg, "dataloader.type") != "piperx_lerobot":
            raise ValueError("OpenPI serving currently requires a piperx_lerobot checkpoint")
        if OmegaConf.select(cfg, "dataloader.action_mode") != "eef":
            raise ValueError("OpenPI Piper-X serving requires action_mode=eef")
        if OmegaConf.select(cfg, "inference.inference_mode", default="sync") != "sync":
            raise ValueError("OpenPI chunks use client-side execution; set --inference-mode sync")
        if OmegaConf.select(cfg, "inference.inference_horizon") is not None:
            raise ValueError("OpenPI returns full chunks; configure policy.n_action_steps on the client")
        self.engine = engine
        self.cfg = cfg
        self.preprocessor = ObsPreprocessor.from_cfg(cfg, engine)
        if not self.preprocessor.multiview or (self.preprocessor.img_height, self.preprocessor.img_width) != (384, 320):
            raise ValueError("OpenPI Piper-X serving requires the trained 384x320 multiview layout")
        if self.preprocessor.camera_layout != [f"observation.images.{name}" for name in VIEWS]:
            raise ValueError("OpenPI Piper-X serving requires the trained front/left/right camera order")
        self.action_horizon = int(cfg.inference.num_frames) - 1
        self.metadata = {
            "server": "openwam-openpi-websocket",
            "protocol_version": 1,
            "view_order": list(VIEWS),
            "state_dim": 14,
            "state_names": JOINT_NAMES,
            "state_representation": "joint_position",
            "joint_position_unit": "radian",
            "action_dim": 20,
            "action_horizon": self.action_horizon,
            "action_representation": "absolute_dual_piper_x_flange_eef20",
            "pose_layout": "[left_xyz,left_rot6d,left_gripper,right_xyz,right_rot6d,right_gripper]",
            "pose_frame": "arm_base",
            "rotation_representation": "rotation_matrix_first_two_columns",
            "position_unit": "meter",
            "gripper_unit": "legacy_piper_radian",
            "action_fps": 30,
            "execution_mode": "client_chunk",
        }

    def infer(self, request: dict) -> dict:
        started = time.perf_counter()
        if not isinstance(request, dict):
            raise ValueError("Observation must be a dict")
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must be finite joint14 in dataset joint order")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty task string")
        images = request.get("images")
        if not isinstance(images, dict):
            raise ValueError("images must contain fixed_front, left_arm and right_arm RGB arrays")
        processed_images = {}
        for name, slot in zip(VIEWS, SLOTS):
            array = np.asarray(images.get(name))
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8 or 0 in array.shape:
                raise ValueError(f"{name} must be nonempty uint8 HWC RGB")
            size = (320, 256) if slot == "head_camera" else (160, 128)
            processed_images[slot] = Image.fromarray(array).resize(size, Image.Resampling.LANCZOS)
        seed = request.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2**63")
        obs = self.preprocessor.preprocess(
            {"images": processed_images, "state": joint14_to_eef20(state), "prompt": prompt}
        )
        conditions = {
            "observation": obs,
            "first_frame_image": [obs["image"]],
            "proprio": obs["state"],
            "prompt": prompt,
            "seed": int(seed),
        }
        infer_started = time.perf_counter()
        result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.action_horizon, 20) or not np.isfinite(actions).all():
            raise RuntimeError(f"Expected finite EEF20 chunk [{self.action_horizon},20], got {actions.shape}")
        return {
            "actions": actions,
            "server_timing": {
                "infer_ms": (time.perf_counter() - infer_started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }

    def run(self, host: str = "0.0.0.0", port: int = 8848):
        import websockets

        async def serve():
            lock = asyncio.Lock()

            async def handler(websocket):
                logger.info("OpenPI client connected: %s", websocket.remote_address)
                try:
                    await websocket.send(msgpack_numpy.packb(self.metadata))
                    async for message in websocket:
                        try:
                            if not isinstance(message, bytes):
                                raise ValueError("OpenPI requests must be binary MessagePack frames")
                            request = msgpack_numpy.unpackb(message)
                            # The adapter has no episode state or action queue to clear.
                            if isinstance(request, dict) and request.get("type") == "reset":
                                response = {"type": "reset_ack"}
                            else:
                                async with lock:
                                    response = await asyncio.to_thread(self.infer, request)
                            await websocket.send(msgpack_numpy.packb(response))
                        except Exception as exc:
                            logger.exception("OpenPI request failed")
                            await websocket.send(f"{type(exc).__name__}: {exc}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    logger.info("OpenPI client disconnected: %s", websocket.remote_address)

            async with websockets.serve(
                handler, host, port, max_size=MAX_MESSAGE_BYTES, compression=None, ping_interval=None
            ):
                logger.info("OpenPI chunk server started on ws://%s:%d", host, port)
                await asyncio.Future()

        asyncio.run(serve())
