"""
Minimal FastAPI server exposing a single LIBERO environment instance.
- No multi-session registry, just ONE env shared by all requests.
- Endpoints to init/change env, step with batches, get state, and reset.
- Episode ends when step_count >= max_steps or env signals done.

Swap `make_libero_env` with your real LIBERO factory call.
"""
from __future__ import annotations

import os
import base64
import io
import httpx
import requests
import numpy as np
import asyncio
from PIL import Image
import uvicorn
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark
from libero.libero import get_libero_path


# ==========================
# Config & Schemas
# ==========================
class InitRequest(BaseModel):
    env_name: str = Field(
        ...,
        description="LIBERO env name e.g., 'libero_object', 'libero_spatial', etc.")
    # Moved SimConfig fields here
    max_steps: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = None
    # Task id to select specific task from the suite
    task_id: int = Field(0, ge=0)
    stop_on_done: bool = True


class InitResponse(BaseModel):
    ok: bool
    env_name: str
    max_steps: int
    initial_image: str


class Action(BaseModel):
    actions: List[float]


class StepBatchRequest(BaseModel):
    # Accept flexible action payloads: single flat list,
    # list of lists, or list of Action objects
    actions: List[Any]
    horizon: int = Field(1, ge=1)
    capture_every: int = Field(1, ge=1)
    stop_on_done: bool = True


class EpisodeBatchRequest(BaseModel):
    instruction: str
    horizon: int = Field(1, ge=1)
    stop_on_done: bool = True
    capture_every: int = Field(1, ge=1)


class ImageItem(BaseModel):
    step_index: int
    image_b64_png: str


class StepBatchResponse(BaseModel):
    images: List[ImageItem]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    total_steps_taken: int
    step_count: int
    done: bool


class EpisodeBatchResponse(BaseModel):
    images: List[ImageItem]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    total_steps_taken: int
    step_count: int
    done: bool


class StateResponse(BaseModel):
    env_name: Optional[str]
    step_count: int
    max_steps: int
    done: bool
    last_info: Optional[Dict[str, Any]] = None
    last_obs: Optional[Dict[str, Any]] = None
    episode_idx: Optional[int] = None
    task_id: Optional[int] = None


# ==========================
# LIBERO Env Glue
# ==========================
ENV_LOCK = asyncio.Lock()
ENV = None
ENV_NAME: Optional[str] = None
TASK_ID = 0
CONFIG = None
STEP_COUNT = 0
TASK_SUITE = None
EPISODE_IDX = 0
DONE = False
LAST_INFO: Optional[Dict[str, Any]] = None
LAST_OBS: Optional[Dict[str, Any]] = None
STOP_ON_DONE = True


# ==========================
# Helpers
# ==========================
def get_max_steps_for_suite(task_suite_name: str) -> int:
    """Get a sensible default max_steps for a given task suite.

    Args:
        task_suite_name (str): libero task suite name

    Returns:
        int: default max steps
    """
    mapping = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    return mapping.get(task_suite_name, 300)


def get_env_info(env_name: str, task_id: int, episode_idx: int) -> Tuple[str, str]:
    """Get environment information for a specific task.

    Args:
        env_name (str): The name of the environment.
        task_id (int): The ID of the task.
        episode_idx (int): The index of the episode.

    Returns:
        Tuple[str, str]: A tuple containing the BDDL file path and task suite name.
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    # Can also choose libero_spatial, libero_object, etc.
    task_suite_name = env_name
    task_suite = benchmark_dict[task_suite_name]()

    # Retrieve a specific task
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder, task.bddl_file
        )

    return task_bddl_file, task_suite_name


def get_env_args(env_name: str, task_id: int, episode_idx: int) -> Dict[str, Any]:
    """Get environment arguments for initializing the LIBERO environment.

    Args:
        env_name (str): The name of the environment.
        task_id (int): The ID of the task.
        episode_idx (int): The index of the episode.

    Returns:
        Dict[str, Any]: A dictionary containing the environment arguments.
    """

    task_bddl_file, task_suite_name = get_env_info(env_name, task_id, episode_idx)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 480,
        "camera_widths": 480
    }
    return env_args


def make_libero_env(env_args):
    """Create a LIBERO environment.

    Args:
        env_args (Dict[str, Any]): Arguments for the environment.

    Returns:
        OffScreenRenderEnv: The created LIBERO environment.
    """
    env = OffScreenRenderEnv(**env_args)
    return env


def encode_png_b64(pil_img: Image.Image) -> str:
    """Encode a PIL Image to a base64 PNG string.

    Args:
        pil_img (Image.Image): The PIL Image to encode.

    Raises:
        TypeError: If the input is not a PIL Image.

    Returns:
        str: The base64 encoded PNG string.
    """
    # Convert numpy array to PIL Image if necessary
    if isinstance(pil_img, np.ndarray):
        arr = pil_img
        # If floats, assume in [0,1] and scale to [0,255]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        pil_img = Image.fromarray(arr)

    if not isinstance(pil_img, Image.Image):
        raise TypeError("encode_png_b64 expects a PIL.Image.Image or a numpy.ndarray")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a given angle.

    Args:
        img (np.ndarray): The input image as a numpy array.
        angle (float): The angle to rotate the image.

    Returns:
        np.ndarray: The rotated image as a numpy array.
    """
    pil_img = Image.fromarray(img)
    rotated = pil_img.rotate(angle)
    return np.array(rotated)


def save_img_to_disk(frame: np.ndarray, step_count: int):
    """Save image frame to disk for debugging.

    Args:
        frame (np.ndarray): The image frame to save.
        step_count (int): The current step count.
    """
    img = Image.fromarray(frame)
    os.makedirs("outputs", exist_ok=True)
    img.save(f"outputs/step_{step_count:04d}.png")


def check_episode_end():
    """Check if the episode has ended.
    """
    global DONE, STEP_COUNT, CONFIG
    if STEP_COUNT >= CONFIG["max_steps"]:
        DONE = True


def skip_steps(env, n: int):
    """Skip a number of steps in the environment.

    Args:
        env (OffScreenRenderEnv): The LIBERO environment.
        n (int): The number of steps to skip.
    """
    # Dummy action of 7 0.0
    dummy_action = [0.0] * 7
    for _ in range(n):
        env.step(dummy_action)  # No-op action


def reset_if_done():
    """Reset the environment if the episode is done.
    """
    global DONE, STEP_COUNT, LAST_INFO, EPISODE_IDX, ENV, TASK_SUITE, TASK_ID
    # advance episode index if multiple init states are available
    initial_states = TASK_SUITE.get_task_init_states(TASK_ID)
    _ = ENV.reset()
    _ = ENV.set_init_state(initial_states[EPISODE_IDX])

    # clear episode counters/state
    STEP_COUNT = 0
    DONE = False
    LAST_INFO = None


async def _apply_reset_common(new_cfg: Optional[Dict[str, Any]] = None,
                              reset_task: bool = False):
    """Common reset logic for the environment.

    Args:
        new_cfg (Optional[Dict[str, Any]], optional): \
            New configuration for the environment. Defaults to None.
        reset_task (bool, optional): Whether to reset the task. Defaults to False.

    Raises:
        HTTPException: If the environment is not initialized.
    """
    global CONFIG, STEP_COUNT, DONE, LAST_INFO, EPISODE_IDX
    if ENV is None:
        raise HTTPException(
            status_code=400,
            detail="Env not initialized. Call /init first."
        )
    # apply optional new config
    if new_cfg is not None:
        if new_cfg.get('max_steps') is None:
            new_max = get_max_steps_for_suite(ENV_NAME)
        else:
            new_max = int(new_cfg.get('max_steps'))
        CONFIG["max_steps"] = new_max
        if 'seed' in new_cfg:
            CONFIG["seed"] = new_cfg.get('seed')

    _ = ENV.reset()
    STEP_COUNT = 0
    DONE = False
    LAST_INFO = None

    if reset_task:
        EPISODE_IDX = 0


class PredictPayload(BaseModel):
    instruction: str
    image: str
    horizon: int


class PredictResponse(BaseModel):
    actions: List[List[float]]


def _encode_image_path_to_b64(path: str) -> str:
    """Encode an image file to a base64 string.

    Args:
        path (str): Path to the image file.

    Returns:
        str: Base64-encoded image string.
    """
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{b64}"


DEFAULT_FALCONVLA_URL = os.getenv("FALCONVLA_URL", "http://localhost:8080")
DEFAULT_FALCONVLA_PREDICT_PATH = "/falconvla/predict"


def falconvla_predict(instruction: str, image_path: str = None, image_b64: str = None, horizon: int = 10, falconvla_url: str = None, timeout: float = 2.0) -> List[List[float]]:
    """Send a prediction request to the FalconVLA API.

    Args:
        instruction (str): The instruction to be processed.
        image_path (str, optional): Path to the image file. Defaults to None.
        image_b64 (str, optional): Base64-encoded image string. Defaults to None.
        horizon (int, optional): Prediction horizon. Defaults to 10.
        falconvla_url (str, optional): Base URL for the FalconVLA API. Defaults to None.
        timeout (float, optional): Timeout for the API request. Defaults to 2.0.

    Raises:
        RuntimeError: If the API request fails.

    Returns:
        List[List[float]]: The predicted actions from the FalconVLA API.
    """
    base = (falconvla_url or DEFAULT_FALCONVLA_URL).rstrip("/")
    url = f"{base}{DEFAULT_FALCONVLA_PREDICT_PATH}"
    if image_b64 is None and image_path is not None:
        image_b64 = _encode_image_path_to_b64(image_path)
    payload = PredictPayload(
        instruction=instruction,
        image=image_b64 or "",
        horizon=horizon,
    )
    try:
        r = requests.post(url, json=payload.model_dump(), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("actions")
    except Exception as e:
        raise RuntimeError(f"FalconVLA predict request failed: {e}")


# ==========================
# FastAPI app
# ==========================
app = FastAPI(title="LIBERO Minimal Server", version="0.2.0")


@app.get("/libero/health")
async def health():
    """Simple health check endpoint."""
    return {"ok": True, "env_initialized": ENV is not None}


# ==========================
# Endpoints
# ==========================
@app.post("/libero/init")
async def init_env(req: InitRequest):
    global ENV, ENV_NAME, TASK_ID, CONFIG, STEP_COUNT
    global DONE, LAST_INFO, TASK_SUITE, EPISODE_IDX
    async with ENV_LOCK:
        # Close previous env if exists
        try:
            if ENV is not None:
                ENV.close()
        except Exception:
            pass

        ENV_NAME = req.env_name
        # Populate runtime config from InitRequest
        CONFIG = dict(max_steps=req.max_steps, seed=req.seed)
        TASK_ID = int(req.task_id)

        # Validate env exists in LIBERO benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        error_message = f"Unknown env_name: {ENV_NAME} " + \
            f"should be one of {list(benchmark_dict.keys())}"
        if ENV_NAME not in benchmark_dict:
            raise HTTPException(
                status_code=400,
                detail=error_message
            )

        # Initialize task suite and episode/task indices
        TASK_SUITE = benchmark_dict[ENV_NAME]()
        # TASK_ID already set from request (defaults to 0)
        EPISODE_IDX = 0

        # If client didn't set max_steps, choose a sensible default
        if (CONFIG["max_steps"] is None or CONFIG["max_steps"] <= 10):
            CONFIG["max_steps"] = get_max_steps_for_suite(ENV_NAME)

        # Build environment args for the selected task and episode
        env_args = get_env_args(ENV_NAME, TASK_ID, EPISODE_IDX)

        ENV = make_libero_env(env_args)

        _ = ENV.reset()

        initial_states = TASK_SUITE.get_task_init_states(TASK_ID)
        _ = ENV.set_init_state(initial_states[EPISODE_IDX])
        # Skip the first 10 steps (the sim drops objects from a high z coordinate)
        skip_steps(ENV, 10)
        obs = ENV.step([0.0]*7)[0]
        # Flip the image 180 degrees for correct orientation
        # LIBERO env camera is upside-down
        initial_image = rotate_image(obs['agentview_image'], 180)
        initial_image = encode_png_b64(initial_image)

        STEP_COUNT = 0
        DONE = False
        LAST_INFO = None
    return {"ok": True, "env_name": ENV_NAME, "max_steps": CONFIG["max_steps"], "initial_image": initial_image}


@app.post("/libero/change_env")
async def change_env(req: InitRequest):
    """Change the environment.

    Args:
        req (InitRequest): The request containing environment parameters.

    Returns:
        Dict: A dictionary indicating success and the new environment details.
    """
    return await init_env(req)


@app.get("/libero/state", response_model=StateResponse)
async def get_state():
    """Get the current state of the environment.

    Returns:
        StateResponse: The current state of the environment.
    """
    global ENV_NAME, STEP_COUNT, CONFIG, DONE
    global LAST_INFO, EPISODE_IDX, TASK_ID
    async with ENV_LOCK:
        return StateResponse(
            env_name=ENV_NAME,
            step_count=STEP_COUNT,
            max_steps=CONFIG["max_steps"],
            done=DONE,
            last_info=LAST_INFO,
            last_obs=LAST_OBS,
            episode_idx=EPISODE_IDX,
            task_id=TASK_ID
        )


@app.post("/libero/reset/episode")
async def reset_episode(new_cfg: Optional[Dict[str, Any]] = None):
    """Reset the current episode in the environment.

    Args:
        new_cfg (Optional[Dict[str, Any]], optional): \
            New configuration for the environment. Defaults to None.

    Returns:
        Dict: A dictionary indicating success and the max steps.
    """
    async with ENV_LOCK:
        await _apply_reset_common(new_cfg=new_cfg, reset_task=False)
    return {"ok": True, "max_steps": CONFIG["max_steps"]}


@app.post("/libero/reset/task")
async def reset_task(new_cfg: Optional[Dict[str, Any]] = None):
    """Reset the task in the environment.

    Args:
        new_cfg (Optional[Dict[str, Any]], optional): \
            New configuration for the environment. Defaults to None.

    Returns:
        Dict: A dictionary indicating success and the max steps.
    """
    async with ENV_LOCK:
        await _apply_reset_common(new_cfg=new_cfg, reset_task=True)
    return {"ok": True, "max_steps": CONFIG["max_steps"]}


@app.post("/libero/run/step", response_model=StepBatchResponse)
async def run_step(req: StepBatchRequest):
    global STEP_COUNT, DONE, LAST_INFO, EPISODE_IDX
    global CONFIG, ENV_NAME, TASK_ID
    images: List[ImageItem] = []
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    total_steps_taken = 0

    # 1) Light-weight validation + optional auto-reset (hold lock briefly)
    async with ENV_LOCK:
        if ENV is None:
            raise HTTPException(status_code=400, detail="Env not initialized. Call /init first.")
        if TASK_SUITE is None:
            raise HTTPException(status_code=400, detail="Task suite not initialized. Call /init first.")
        if STOP_ON_DONE and DONE:
            reset_if_done()
        # capture context to send to FalconVLA (do not hold lock while calling external service)
        ctx_info = LAST_INFO

    # 2) Determine actions sequence: either client-provided or fetched from FalconVLA
    actions_sequence = req.actions
    if len(actions_sequence) > 0 and isinstance(actions_sequence[0], (int, float)):
        actions_sequence = [actions_sequence]

    if not actions_sequence:
        # fetch from FalconVLA (network IO) without holding ENV_LOCK
        actions_sequence = await falconvla_predict(req.horizon, ctx_info)

    # Ensure there are at least 'horizon' actions
    if len(actions_sequence) < req.horizon:
        if len(actions_sequence) == 0:
            actions_sequence = [[0.0] * 7] * req.horizon
        else:
            actions_sequence = (actions_sequence * ((req.horizon // len(actions_sequence)) + 1))[:req.horizon]

    # 3) Execute steps; acquire ENV_LOCK only around the env.step() + state updates
    for action_idx in range(req.horizon):
        if DONE:
            break

        action = actions_sequence[action_idx]

        async with ENV_LOCK:
            out = ENV.step(action)
            obs, reward, done, info = out

            STEP_COUNT += 1
            total_steps_taken += 1
            LAST_INFO = info
            rewards.append(float(reward))
            infos.append(info)

            if (STEP_COUNT % req.capture_every) == 0:
                frame = obs['agentview_image']
                frame = rotate_image(frame, 180)
                save_img_to_disk(frame, STEP_COUNT)
                images.append(
                    ImageItem(
                        step_index=STEP_COUNT,
                        image_b64_png=encode_png_b64(frame)
                    )
                )

            # termination checks
            DONE = DONE or bool(done)
            check_episode_end()
            if req.stop_on_done and DONE:
                break

    return StepBatchResponse(
        images=images,
        rewards=rewards,
        infos=infos,
        total_steps_taken=total_steps_taken,
        step_count=STEP_COUNT,
        done=DONE,
    )


@app.post("/libero/run/episode", response_model=EpisodeBatchResponse)
async def run_episode(req: EpisodeBatchRequest):
    global STEP_COUNT, DONE, LAST_INFO, LAST_OBS
    images: List[ImageItem] = []
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    total_steps_taken = 0
    instruction = req.instruction

    async with ENV_LOCK:
        global TASK_ID, TASK_SUITE, EPISODE_IDX

        if ENV is None:
            raise HTTPException(
                status_code=400,
                detail="Env not initialized. Call /init first."
                )

        # Ensure task suite is valid
        if TASK_SUITE is None:
            raise HTTPException(
                status_code=400,
                detail="Task suite not initialized. Call /init first."
            )

        # If episode ended and client asked to auto-reset,
        # do it here (still under ENV_LOCK)
        if STOP_ON_DONE and DONE:
            reset_if_done()

        obs = ENV.step([0.0]*7)[0]  # Dummy step to get current obs
        # Loop through the episode max steps
        for ep_step in range(CONFIG["max_steps"]):
            # Get the action from FalconVLA without holding ENV_LOCK
            image_b64 = encode_png_b64(
                rotate_image(obs['agentview_image'], 180)
                )
            actions_sequence = falconvla_predict(
                instruction=instruction,
                image_b64=image_b64
            )
            # Loop through the action sequence
            for action_idx in range(req.horizon):
                if DONE:
                    break
                # One step in the LIBERO env
                out = ENV.step(actions_sequence[action_idx])
                # Unpack the output of the env after the step
                obs, reward, done, info = out

                STEP_COUNT += 1
                total_steps_taken += 1
                LAST_INFO = info
                rewards.append(float(reward))
                infos.append(info)

                if (STEP_COUNT % req.capture_every) == 0:
                    frame = obs['agentview_image']
                    # Flip the image 180 degrees for correct orientation
                    # LIBERO env camera is upside-down
                    frame = rotate_image(frame, 180)
                    save_img_to_disk(frame, STEP_COUNT)
                    images.append(
                        ImageItem(
                            step_index=STEP_COUNT,
                            image_b64_png=encode_png_b64(frame)
                            )
                        )

                # termination checks
                DONE = DONE or bool(done)
                check_episode_end()
                if req.stop_on_done and DONE:
                    break

        return EpisodeBatchResponse(
            images=images,
            rewards=rewards,
            infos=infos,
            total_steps_taken=total_steps_taken,
            step_count=STEP_COUNT,
            done=DONE,
        )


# Run: uvicorn server_single:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8088, help="Port to run the server on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
