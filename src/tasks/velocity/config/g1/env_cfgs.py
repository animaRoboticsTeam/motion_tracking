"""Unitree G1 velocity environment configurations."""

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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

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
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

def unitree_g1_lower_body_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 lower-body and waist flat terrain locomotion and balance configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  lower_body_joints = (
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
  cfg.actions["joint_pos"].actuator_names = lower_body_joints
  if isinstance(cfg.actions["joint_pos"].scale, dict):
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in cfg.actions["joint_pos"].scale.items()
        if not any(x in k for x in ("elbow", "shoulder", "wrist"))
    }

  # Align joint_names for posture rewards to match lower body dimension (15).
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].params["asset_cfg"].joint_names = lower_body_joints
    for std_key in ("std_walking", "std_running"):
      if std_key in cfg.rewards["pose"].params:
        old_std = cfg.rewards["pose"].params[std_key]
        cfg.rewards["pose"].params[std_key] = {
            k: v for k, v in old_std.items()
            if not any(arm in k for arm in ("shoulder", "elbow", "wrist"))
        }

  if "stand_still" in cfg.rewards:
    cfg.rewards["stand_still"].params["asset_cfg"].joint_names = lower_body_joints

  # Limit joint_pos and joint_vel observations to only lower body joints
  for group_name in ("actor", "critic"):
    if group_name in cfg.observations:
      group = cfg.observations[group_name]
      if "joint_pos" in group.terms:
        group.terms["joint_pos"].params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=lower_body_joints)
        }
      if "joint_vel" in group.terms:
        group.terms["joint_vel"].params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=lower_body_joints)
        }

  if not play:
    # 1. Increase CoM randomization range to simulate upper body movement/payload shifts.
    if "base_com" in cfg.events:
      cfg.events["base_com"].params["ranges"] = {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
      }

    # 2. Add roll and pitch perturbation at reset to force the robot to learn balance recovery from tilted states.
    if "reset_base" in cfg.events:
      cfg.events["reset_base"].params["pose_range"] = {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-3.14, 3.14),
      }

    # 3. Relax posture and stand_still penalties so the robot is willing to step/move legs to recover balance when standing/moving slowly.
    if "pose" in cfg.rewards:
      # Loosen std_standing from 0.05 to allow more leg movement during balance recovery
      cfg.rewards["pose"].params["std_standing"] = {
          r".*hip_pitch.*": 0.3,
          r".*hip_roll.*": 0.2,
          r".*hip_yaw.*": 0.2,
          r".*knee.*": 0.3,
          r".*ankle_pitch.*": 0.2,
          r".*ankle_roll.*": 0.15,
          r".*waist.*": 0.15,
      }
    if "stand_still" in cfg.rewards:
      cfg.rewards["stand_still"].weight = -0.05

  if play:
    # twist_cmd = cfg.commands["twist"]
    # assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # # 将命令速度随机范围强行修改为 [0.0, 0.0]
    # twist_cmd.ranges.lin_vel_x = (0.0, 0.0)  # 原为 (-0.5, 1.0)
    # twist_cmd.ranges.lin_vel_y = (0.0, 0.0)  # 原为 (-0.5, 0.5)
    # twist_cmd.ranges.ang_vel_z = (0.0, 0.0)  # 原为 (-0.5, 0.5)

    cfg.events["randomize_arm_joints"] = EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(".*shoulder.*", ".*elbow.*", ".*wrist.*")
            ),
        },
    )

  return cfg