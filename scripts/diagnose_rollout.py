"""Diagnostic: what does the checkpoint actually output in one episode?"""

import sys

import numpy as np

sys.path.insert(0, "/home/xy/Embodied_AI_Try/src")

from evla.envs.registry import make_env
from evla.models.registry import load_policy

ckpt = sys.argv[1]
env = make_env({"name": "pusht", "max_steps": 60})
policy = load_policy({"type": "smolvla", "checkpoint": ckpt, "device": "cuda",
                      "precision": "bf16", "image_key": "top"})
obs = env.reset(seed=0)
print("task:", obs.task)
print("state:", obs.state, "image:", obs.images["top"].shape, obs.images["top"].dtype)
chunk = policy.predict_chunk(obs)
print("chunk shape", chunk.shape)
print("first 5 actions:\n", np.round(chunk[:5], 1))
print("action min/max", chunk.min(0).round(1), chunk.max(0).round(1))
for i in range(12):
    a = chunk[i]
    res = env.step(a)
    info = res.info
    print(f"step {i} action {a.round(1)} agent {obs.state[:2]} -> {res.obs.state[:2]} success={res.success}")
    if res.done:
        break
env.close()
