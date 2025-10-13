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
class SimConfig(BaseModel):
    max_steps: Optional[int] = Field(None, ge=1)
    seed: Optional[int] = None


class InitRequest(BaseModel):
    env_name: str = Field(
        ...,
        description="LIBERO env name e.g., \
            'libero_object', 'libero_spatial', etc.")
    config: SimConfig = Field(default_factory=SimConfig)


class Action(BaseModel):
    actions: List[float]


class StepBatchRequest(BaseModel):
    actions: List[float]
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


class StateResponse(BaseModel):
    env_name: Optional[str]
    step_count: int
    max_steps: int
    done: bool
    last_info: Optional[Dict[str, Any]] = None


# ==========================
# LIBERO Env Glue
# ==========================
ENV_LOCK = asyncio.Lock()
ENV = None
ENV_NAME: Optional[str] = None
TASK_ID = 0
CONFIG = SimConfig()
STEP_COUNT = 0
TASK_SUITE = None
EPISODE_IDX = 0
DONE = False
LAST_INFO: Optional[Dict[str, Any]] = None


# ==========================
# Helpers
# ==========================
def get_max_steps_for_suite(task_suite_name: str) -> int:
    """Return an appropriate max_steps value for known
    LIBERO suites."""
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
    task_bddl_file, task_suite_name = get_env_info(env_name, task_id, episode_idx)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    return env_args


def make_libero_env(env_args):
    env = OffScreenRenderEnv(**env_args)
    return env


def encode_png_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_frame_from_env(env) -> Image.Image:
    img = env.render(mode="rgb_array")
    if isinstance(img, Image.Image):
        return img
    # If numpy array:
    try:
        return Image.fromarray(img)
    except Exception:
        raise RuntimeError("Unable to obtain image frame from env.render()")


def to_action_payload(a: List[float]) -> List[float]:
    """Convert API action list to env action payload."""
    # Action model uses `actions` field (list[float]).
    # Return it directly as the payload for the LIBERO env.
    return a


def check_episode_end():
    global DONE, STEP_COUNT, CONFIG
    if STEP_COUNT >= CONFIG.max_steps:
        DONE = True


# ==========================
# FastAPI app
# ==========================
app = FastAPI(title="LIBERO Minimal Server", version="0.2.0")


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
        CONFIG = req.config

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
        TASK_ID = 0
        EPISODE_IDX = 0

        # If client didn't set max_steps, choose a sensible default
        if (CONFIG.max_steps is None or CONFIG.max_steps <= 10):
            CONFIG.max_steps = get_max_steps_for_suite(ENV_NAME)

        # Build environment args for the selected task and episode
        env_args = get_env_args(ENV_NAME, TASK_ID, EPISODE_IDX)

        ENV = make_libero_env(env_args)

        _ = ENV.reset()
        STEP_COUNT = 0
        DONE = False
        LAST_INFO = None
    return {"ok": True, "env_name": ENV_NAME, "max_steps": CONFIG.max_steps}


@app.post("/change_env")
async def change_env(req: InitRequest):
    return await init_env(req)


@app.get("/state", response_model=StateResponse)
async def get_state():
    async with ENV_LOCK:
        return StateResponse(
            env_name=ENV_NAME,
            step_count=STEP_COUNT,
            max_steps=CONFIG.max_steps,
            done=DONE,
            last_info=LAST_INFO,
        )


@app.post("/reset")
async def reset_env(new_cfg: Optional[SimConfig] = None):
    global CONFIG, STEP_COUNT, DONE, LAST_INFO
    async with ENV_LOCK:
        if ENV is None:
            raise HTTPException(
                status_code=400,
                detail="Env not initialized. Call /init first."
            )
        if new_cfg is not None:
            # If client didn't set max_steps, choose a sensible default
            if new_cfg.max_steps is None:
                # ENV_NAME should be set because ENV exists
                new_cfg.max_steps = get_max_steps_for_suite(ENV_NAME)
            CONFIG = new_cfg
        _ = ENV.reset()
        STEP_COUNT = 0
        DONE = False
        LAST_INFO = None
    return {"ok": True, "max_steps": CONFIG.max_steps}


@app.post("/actions", response_model=StepBatchResponse)
async def step_actions(req: StepBatchRequest):
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

        # Ensure task suite and episode index are valid
        if TASK_SUITE is None:
            raise HTTPException(
                status_code=400,
                detail="Task suite not initialized. Call /init first."
            )

        # Set initial states
        initial_states = TASK_SUITE.get_task_init_states(TASK_ID)
        if EPISODE_IDX < 0 or EPISODE_IDX >= len(initial_states):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid episode index {EPISODE_IDX} for task {TASK_ID}."
                ),
            )
        obs = ENV.set_init_state(initial_states[EPISODE_IDX])

        # Step through all actions
        for a in req.actions:
            for _ in range(req.horizon):
                if DONE:
                    break
                action_payload = to_action_payload(a)
                # LIBERO API variants: (obs, reward, done, info)
                out = ENV.step(action_payload)
                obs, reward, done, info = out

                STEP_COUNT += 1
                total_steps_taken += 1
                LAST_INFO = info
                rewards.append(float(reward))
                infos.append(info)

                if (STEP_COUNT % req.capture_every) == 0:
                    frame = get_frame_from_env(ENV)
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
