from mjlab.tasks.registry import register_mjlab_task
from src.tasks.squat.rl.runner import SquatOnPolicyRunner

from .env_cfgs import (
  unitree_g1_squat_flat_env_cfg,
  unitree_g1_squat_lower_body_flat_env_cfg,
  unitree_g1_squat_lower_body_rough_env_cfg,
  unitree_g1_squat_rough_env_cfg,
)
from .rl_cfg import unitree_g1_squat_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Squat-Rough",
  env_cfg=unitree_g1_squat_rough_env_cfg(),
  play_env_cfg=unitree_g1_squat_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_squat_ppo_runner_cfg(),
  runner_cls=SquatOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Squat-Flat",
  env_cfg=unitree_g1_squat_flat_env_cfg(),
  play_env_cfg=unitree_g1_squat_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_squat_ppo_runner_cfg(),
  runner_cls=SquatOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Squat-Rough-LowerBody",
  env_cfg=unitree_g1_squat_lower_body_rough_env_cfg(),
  play_env_cfg=unitree_g1_squat_lower_body_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_squat_ppo_runner_cfg(),
  runner_cls=SquatOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Squat-Flat-LowerBody",
  env_cfg=unitree_g1_squat_lower_body_flat_env_cfg(),
  play_env_cfg=unitree_g1_squat_lower_body_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_squat_ppo_runner_cfg(),
  runner_cls=SquatOnPolicyRunner,
)
