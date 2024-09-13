# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class TriStageLRScheduleConfig(FairseqDataclass):
    warmup_steps: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    hold_steps: int = field(
        default=0,
        metadata={"help": "steps in hold stage"},
    )
    decay_steps: int = field(
        default=0,
        metadata={"help": "steps in decay stages"},
    )
    phase_ratio: Optional[Tuple[float, float, float]] = field(
        default=None,
        metadata={
            "help": (
                "if set, automatically sets warmup/hold/decay steps to the ratio "
                "specified here from max_updates. the ratios must add up to 1.0"
            )
        },
    )
    init_lr_scale: float = field(
        default=0.01,
        metadata={"help": "initial learning rate scale during warmup phase"},
    )
    final_lr_scale: float = field(
        default=0.01,
        metadata={"help": "final learning rate scale"},
    )
    max_update: float = II("optimization.max_update")
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("tri_stage", dataclass=TriStageLRScheduleConfig)
class TriStageLRSchedule(FairseqLRScheduler):
    """Tristage learning rate schedulr

    Implement the learning rate scheduler in https://arxiv.org/pdf/1904.08779.pdf

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations

        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations

        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::

      init_lr = cfg.init_lr_scale * cfg.lr
      lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
      lr = lrs[update_num]

    During hold::

      lr = cfg.lr

    During decay::

      decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
      lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

    After that::

      lr = cfg.lr * cfg.final_lr_scale
    """

    def __init__(self, cfg: TriStageLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with tri-stage lr."
                " Consider --lr-scheduler=fixed instead."
            )

        # calculate LR at each point
        self.peak_lr = cfg.lr[0]
        self.init_lr = cfg.init_lr_scale * cfg.lr[0]
        self.final_lr = cfg.final_lr_scale * cfg.lr[0]

        if cfg.phase_ratio is not None:
            assert cfg.max_update > 0
            assert sum(cfg.phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(cfg.max_update * cfg.phase_ratio[0])
            self.hold_steps = int(cfg.max_update * cfg.phase_ratio[1])
            self.decay_steps = int(cfg.max_update * cfg.phase_ratio[2])
        else:
            self.warmup_steps = cfg.warmup_steps
            self.hold_steps = cfg.hold_steps
            self.decay_steps = cfg.decay_steps

        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(cfg.final_lr_scale) / self.decay_steps

        # initial learning rate
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

        self.step_increase = 0
        self.step_decrease = 0
        self.last_num_update = 0
        self.val_loss = None
        self.prune_mode = -1
        self.maxlr = self.final_lr

    def _decide_stage(self, update_step, val_loss):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates, val_loss=None):
        """Update the learning rate after each update."""
        # stage, steps_in_stage = self._decide_stage(num_updates, val_loss)

        if val_loss is not None:
            self.prune_mode = val_loss
            diff_updates = num_updates - self.last_num_update
            self.last_num_update = num_updates

            if self.prune_mode is 1:
                # warmup phase
                self.lr = min(self.peak_lr, self.init_lr + self.warmup_rate * num_updates)
            elif self.prune_mode is 2:
                # increase lr slowly
                # if num_updates / 1101 <= 30:
                #     lr_upperbound = 2 * self.peak_lr
                # else:
                #     lr_upperbound = 3 * self.peak_lr

                self.lr = min(10 * self.peak_lr, self.lr + self.warmup_rate * diff_updates)
            elif self.prune_mode is 3:
                # decrease lr slowly
                self.lr = max(self.peak_lr, self.lr - self.warmup_rate * diff_updates)

            # if self.prune_mode is not 4:
            #     self.maxlr = self.lr
            #     self.decay_factor = -math.log(self.final_lr/self.maxlr) / self.decay_steps

            if self.prune_mode is 4:
                # decay phase
                self.step_decrease += diff_updates
                # self.lr = max(self.final_lr, self.maxlr * math.exp(-self.decay_factor * self.step_decrease))
                self.lr = max(self.final_lr, self.peak_lr * math.exp(-self.decay_factor * self.step_decrease))

            print('prune mode is', self.prune_mode)
            print('self lr is', self.lr)




        # diff_updates = num_updates - self.last_num_update
        # self.last_num_update = num_updates
        # if val_loss is not None:
        #     self.val_loss = val_loss
        #
        # if self.val_loss is None or self.val_loss < 0:
        #     self.step_decrease = 0
        #     self.step_increase += diff_updates
        #     self.lr = min(self.peak_lr, self.init_lr + self.warmup_rate * self.step_increase)
        # else:
        #     self.step_increase = 0
        #     self.step_decrease += diff_updates
        #     self.lr = max(self.final_lr, self.peak_lr * math.exp(-self.decay_factor * self.step_decrease))

        # if stage == 0:
        #     self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        # elif stage == 1:
        #     self.lr = self.peak_lr
        # elif stage == 2:
        #     self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        # elif stage == 3:
        #     self.lr = self.final_lr
        # else:
        #     raise ValueError("Undefined stage")

        self.optimizer.set_lr(self.lr)

        return self.lr
