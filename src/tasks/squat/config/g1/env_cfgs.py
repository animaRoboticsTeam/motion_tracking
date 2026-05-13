"""Unitree G1 squat environment configurations."""

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

import src.tasks.squat.mdp as mdp
from src.tasks.squat.squat_env_cfg import make_squat_env_cfg


def unitree_g1_squat_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain squat configuration."""
  cfg = make_squat_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False  # Keep terrain curriculum flat/simple during balance tests.

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  cfg.observations["actor"].terms["base_height"].params["asset_cfg"].site_names = site_names
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Configure random upper-body perturbations to train balance resilience.
  cfg.events["upper_body_perturb"] = EventTermCfg(
    mode="interval",
    interval_range_s=(1.5, 4.0),
    func=velocity_mdp.push_by_setting_velocity,
  )
  # Let's pop the placeholder to prevent unnecessary run errors, and use com & friction static randomizations.
  cfg.events.pop("upper_body_perturb", None)

  cfg.rewards["pose"].params["stds"] = {
    r".*hip_pitch.*": 0.25,
    r".*hip_roll.*": 0.10,
    r".*hip_yaw.*": 0.10,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.10,
    r".*waist_yaw.*": 0.10,
    r".*waist_roll.*": 0.10,
    r".*waist_pitch.*": 0.10,
    r".*shoulder.*": 0.10,
    r".*elbow.*": 0.10,
    r".*wrist.*": 0.10,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events = {}

  return cfg


def unitree_g1_squat_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain squat configuration."""
  cfg = unitree_g1_squat_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  return cfg


def unitree_g1_squat_lower_body_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 lower-body and waist rough terrain squat configuration."""
  cfg = unitree_g1_squat_rough_env_cfg(play=play)
  cfg.actions["joint_pos"].actuator_names = (
      r".*hip_pitch_joint",
      r".*hip_roll_joint",
      r".*hip_yaw_joint",
      r".*knee_joint",
      r".*ankle_pitch_joint",
      r".*ankle_roll_joint",
      r"waist_yaw_joint",
      r"waist_pitch_joint",
      r"waist_roll_joint",
  )
  if isinstance(cfg.actions["joint_pos"].scale, dict):
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in cfg.actions["joint_pos"].scale.items()
        if not any(x in k for x in ("elbow", "shoulder", "wrist"))
    }
  if isinstance(cfg.rewards["pose"].params["stds"], dict):
    cfg.rewards["pose"].params["stds"] = {
        k: v for k, v in cfg.rewards["pose"].params["stds"].items()
        if not any(x in k for x in ("elbow", "shoulder", "wrist"))
    }
  return cfg


def unitree_g1_squat_lower_body_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 lower-body and waist flat terrain squat configuration."""
  cfg = unitree_g1_squat_flat_env_cfg(play=play)
  cfg.actions["joint_pos"].actuator_names = (
      r".*hip_pitch_joint",
      r".*hip_roll_joint",
      r".*hip_yaw_joint",
      r".*knee_joint",
      r".*ankle_pitch_joint",
      r".*ankle_roll_joint",
      r"waist_yaw_joint",
      r"waist_pitch_joint",
      r"waist_roll_joint",
  )
  if isinstance(cfg.actions["joint_pos"].scale, dict):
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in cfg.actions["joint_pos"].scale.items()
        if not any(x in k for x in ("elbow", "shoulder", "wrist"))
    }
  if isinstance(cfg.rewards["pose"].params["stds"], dict):
    cfg.rewards["pose"].params["stds"] = {
        k: v for k, v in cfg.rewards["pose"].params["stds"].items()
        if not any(x in k for x in ("elbow", "shoulder", "wrist"))
    }
  return cfg
