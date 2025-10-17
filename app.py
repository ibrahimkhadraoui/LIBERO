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


class Action(BaseModel):
    actions: List[float]


class StepBatchRequest(BaseModel):
    # Accept flexible action payloads: single flat list,
    # list of lists, or list of Action objects
    actions: List[Any]
    horizon: int = Field(1, ge=1)
    capture_every: int = Field(1, ge=1)


class EpisodeBatchRequest(BaseModel):
    capture_every: int = Field(1, ge=1)


class ImageItem(BaseModel):
    step_index: int
    image_b64_png: str


class StepBatchResponse(BaseModel):
    images: List[ImageItem]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    obs: Optional[Dict[str, Any]] = None
    total_steps_taken: int
    step_count: int
    done: bool


class EpisodeBatchResponse(BaseModel):
    images: List[ImageItem]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    obs: Optional[Dict[str, Any]] = None
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
        "camera_heights": 128,
        "camera_widths": 128
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


async def fetch_actions_from_falconvla(instruction: str,
                                       obs: Optional[Dict[str, Any]] = None
                                       ) -> List[List[float]]:
    """Fetch actions from FalconVLA service.

    Args:
        instruction (str): The instruction for the action.
        obs (Optional[Dict[str, Any]], optional): Observation information for the request. Defaults to None.

    Returns:
        List[List[float]]: The predicted action chunk.
    """
    url = os.getenv("FALCONVLA_URL", "http://localhost:8080/predict")
    timeout = float(os.getenv("FALCONVLA_TIMEOUT", "1.0"))
    default_action = [0.0] * 7

    # build image payload if observation contains agent view
    image_b64 = None
    if obs is not None:
        try:
            frame = obs.get("agentview_image") if isinstance(obs, dict) else None
            if frame is not None:
                image_b64 = encode_png_b64(frame)
        except Exception:
            image_b64 = None

    payload = {"instruction": instruction, "image": image_b64 or ""}

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            actions = data.get("actions")
            # Basic validation / normalization
            if not actions or not isinstance(actions, list):
                return [default_action]
            # If returned a flat numeric action, wrap it
            if isinstance(actions[0], (int, float)):
                actions = [actions]
            # Ensure each action is a list of floats
            normalized: List[List[float]] = []
            for a in actions:
                if isinstance(a, (list, tuple)):
                    normalized.append([float(x) for x in a])
                else:
                    normalized.append([float(a)])
            return normalized
        except Exception:
            return [default_action]


# ==========================
# FastAPI app
# ==========================
app = FastAPI(title="LIBERO Minimal Server", version="0.2.0")


# ==========================
# Endpoints
# ==========================
@app.post("/init")
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

        STEP_COUNT = 0
        DONE = False
        LAST_INFO = None
    return {"ok": True, "env_name": ENV_NAME, "max_steps": CONFIG["max_steps"]}


@app.post("/change_env")
async def change_env(req: InitRequest):
    """Change the environment.

    Args:
        req (InitRequest): The request containing environment parameters.

    Returns:
        Dict: A dictionary indicating success and the new environment details.
    """
    return await init_env(req)


@app.get("/state", response_model=StateResponse)
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


@app.post("/reset/episode")
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


@app.post("/reset/task")
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


@app.post("/run/step", response_model=EpisodeBatchResponse)
async def run_step(req: EpisodeBatchRequest):
    global STEP_COUNT, DONE, LAST_INFO
    images: List[ImageItem] = []
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    total_steps_taken = 0

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

        # Loop through the episode max steps
        for ep_step in range(CONFIG["max_steps"]):
            # Get the action from FalconVLA without holding ENV_LOCK
            actions_sequence = await fetch_actions_from_falconvla(
                instruction="Complete the task as efficiently as possible.",
                obs=LAST_INFO
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


@app.post("/run/episode", response_model=EpisodeBatchResponse)
async def run_episode(req: StepBatchRequest):
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
        actions_sequence = await fetch_actions_from_falconvla(req.horizon, ctx_info)

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


# Run: uvicorn server_single:app --host 0.0.0.0 --port 8000 --reload
uvicorn.run(app, host="0.0.0.0", port=8123)
