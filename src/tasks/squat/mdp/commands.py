from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformHeightCommand(CommandTerm):
  cfg: UniformHeightCommandCfg

  def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class UniformHeightCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> UniformHeightCommand:
    return UniformHeightCommand(self, env)
