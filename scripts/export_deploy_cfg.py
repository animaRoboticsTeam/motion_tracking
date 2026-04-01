from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv


class _InlineListDumper(yaml.SafeDumper):
    """强制将所有列表序列化为 YAML 行内风格（[a, b, c]）。"""


def _represent_list_inline(dumper: _InlineListDumper, data: list[Any]) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_InlineListDumper.add_representer(list, _represent_list_inline)


def _to_plain_value(value: Any) -> Any:
    """将 torch/tuple/自定义对象递归转换为可 YAML 序列化的基础类型。"""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return _to_plain_value(value.detach().cpu().tolist())
    if isinstance(value, tuple):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_plain_value(v) for k, v in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return _to_plain_value(vars(value))
    if isinstance(value, float):
        # 与参考 deploy.yaml 风格对齐：
        # - 绝大多数 scale 在 (0, 1) 区间，保留两位小数；
        # - 其余较大数值（如 stiffness/damping）保留一位小数。
        if abs(value) < 1.0:
            return float(f"{value:.2f}")
        return float(f"{value:.1f}")
    return value


def _obs_export_name(train_name: str, params: dict[str, Any]) -> str:
    """将训练侧观测名映射到部署端注册的观测名。"""
    if train_name == "command":
        cmd_name = params.get("command_name")
        if cmd_name == "twist":
            return "velocity_commands"
        if cmd_name == "motion":
            return "motion_command"
    if train_name == "phase":
        return "gait_phase"
    if train_name == "joint_pos":
        return "joint_pos_rel"
    if train_name == "joint_vel":
        return "joint_vel_rel"
    if train_name == "actions":
        return "last_action"
    return train_name


def _obs_export_params(train_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """修正部署侧真正使用的参数键值，避免训练期命名差异影响部署。"""
    out = dict(params)
    if train_name == "command" and out.get("command_name") == "twist":
        out["command_name"] = "base_velocity"
    if train_name == "phase":
        out = {"period": out.get("period", 0.6)}
    if train_name in {"joint_pos", "joint_vel", "actions"}:
        out = {}
    return out


def _build_joint_pd_from_cfg(env: ManagerBasedRlEnv) -> tuple[list[float], list[float]]:
    """从机器人 articulation 配置恢复每个仿真关节的刚度和阻尼。"""
    robot = env.scene["robot"]
    num_joints = len(robot.joint_names)
    stiffness = [0.0] * num_joints
    damping = [0.0] * num_joints

    # actuator 运行时对象记录了实际匹配到的 joint id；其 cfg 提供对应 PD 参数。
    for actuator in robot.actuators:
        cfg = actuator.cfg
        joint_ids = actuator._target_ids.tolist()  # noqa: SLF001
        for jid in joint_ids:
            stiffness[jid] = float(cfg.stiffness)
            damping[jid] = float(cfg.damping)
    return stiffness, damping


def export_deploy_cfg(env: ManagerBasedRlEnv, log_dir: Path):
    """从训练环境导出部署端所需 deploy.yaml。"""
    output_path = Path(log_dir)
    if output_path.suffix.lower() != ".yaml":
        output_path = output_path / "params" / "deploy.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot = env.scene["robot"]

    # 当前项目部署执行链默认按动作维度顺序写入关节命令。
    action_term = next(iter(env.action_manager._terms.values()))
    target_ids = action_term.target_ids.detach().cpu().tolist()
    joint_ids_map = [int(i) for i in target_ids]

    stiffness_sim, damping_sim = _build_joint_pd_from_cfg(env)
    default_joint_pos_sim = (
        robot.data.default_joint_pos[0].detach().cpu().tolist()
    )

    # deploy 侧期望按 SDK 关节索引排列。
    num_sdk_joints = max(joint_ids_map) + 1 if joint_ids_map else len(stiffness_sim)
    stiffness = [0.0] * num_sdk_joints
    damping = [0.0] * num_sdk_joints
    default_joint_pos = [0.0] * num_sdk_joints
    for action_idx, sdk_idx in enumerate(joint_ids_map):
        sim_idx = int(target_ids[action_idx])
        stiffness[sdk_idx] = float(stiffness_sim[sim_idx])
        damping[sdk_idx] = float(damping_sim[sim_idx])
        default_joint_pos[sdk_idx] = float(default_joint_pos_sim[sim_idx])

    cfg: dict[str, Any] = {
        "joint_ids_map": joint_ids_map,
        "step_dt": float(env.step_dt),
        "stiffness": stiffness,
        "damping": damping,
        "default_joint_pos": default_joint_pos,
    }

    # 速度任务导出命令范围；模仿任务保持空字典，与现有示例一致。
    commands: dict[str, Any] = {}
    if "twist" in env.cfg.commands:
        cmd_cfg = env.cfg.commands["twist"]
        ranges_cfg = getattr(cmd_cfg, "ranges", None)
        if ranges_cfg is None:
            raise ValueError("twist command config missing 'ranges', cannot export deploy commands")
        ranges = {
            "lin_vel_x": list(ranges_cfg.lin_vel_x),
            "lin_vel_y": list(ranges_cfg.lin_vel_y),
            "ang_vel_z": list(ranges_cfg.ang_vel_z),
            "heading": None,
        }
        commands["base_velocity"] = {"ranges": ranges}
    cfg["commands"] = commands

    # 动作项
    cfg["actions"] = {}
    for term in env.action_manager._terms.values():
        action_name = term.__class__.__name__
        term_cfg = term.cfg
        action_dim = int(term.action_dim)

        scale = term._scale[0].detach().cpu().tolist()  # noqa: SLF001
        offset = term._offset[0].detach().cpu().tolist()  # noqa: SLF001
        clip = getattr(term_cfg, "clip", None)

        cfg["actions"][action_name] = {
            "clip": _to_plain_value(clip),
            "joint_names": list(getattr(term_cfg, "actuator_names", (".*",))),
            "scale": scale if isinstance(scale, list) else [float(scale)] * action_dim,
            "offset": offset if isinstance(offset, list) else [float(offset)] * action_dim,
            "joint_ids": None,
        }

    # 观测项：优先导出演员网络输入组（actor/policy）。
    obs_group_name = "policy"
    if obs_group_name not in env.observation_manager.active_terms:
        obs_group_name = "actor"
    if obs_group_name not in env.observation_manager.active_terms:
        obs_group_name = next(iter(env.observation_manager.active_terms.keys()))

    obs_names = env.observation_manager.active_terms[obs_group_name]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs[obs_group_name]
    cfg["observations"] = {}

    for train_name, obs_cfg in zip(obs_names, obs_cfgs, strict=True):
        params = dict(obs_cfg.params)
        export_name = _obs_export_name(train_name, params)
        export_params = _obs_export_params(train_name, params)

        obs_sample = obs_cfg.func(env, **params)
        obs_dim = int(obs_sample.shape[1]) if obs_sample.ndim > 1 else int(obs_sample.shape[0])

        scale = obs_cfg.scale
        if scale is None:
            scale_list = [1.0] * obs_dim
        else:
            plain_scale = _to_plain_value(scale)
            if isinstance(plain_scale, list):
                scale_list = plain_scale
            else:
                scale_list = [float(plain_scale)] * obs_dim

        clip = _to_plain_value(obs_cfg.clip)
        if clip is not None and not isinstance(clip, list):
            clip = list(clip)

        history_length = int(obs_cfg.history_length) if obs_cfg.history_length else 1
        cfg["observations"][export_name] = {
            "params": _to_plain_value(export_params),
            "clip": clip,
            "scale": _to_plain_value(scale_list),
            "history_length": history_length,
        }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            _to_plain_value(cfg),
            f,
            Dumper=_InlineListDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=False,
            width=120,
        )
