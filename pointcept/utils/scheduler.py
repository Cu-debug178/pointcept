"""
Scheduler

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[int(rate * total_steps) for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        power=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        gamma=0.9,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer,
        total_steps,
        eta_min=0,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )


@SCHEDULERS.register_module()
class StandaloneS3DISLR(lr_scheduler.LambdaLR):
    """KPConvX Standalone S3DIS schedule expressed in optimizer steps."""

    def __init__(
        self,
        optimizer,
        total_steps,
        start_lr=1e-4,
        peak_lr=5e-3,
        reference_epochs=450,
        warmup_epochs=30,
        plateau_epochs=5,
        decay10_epochs=120,
        last_epoch=-1,
    ):
        total_steps = int(total_steps)
        reference_epochs = float(reference_epochs)
        warmup_epochs = float(warmup_epochs)
        plateau_epochs = float(plateau_epochs)
        decay10_epochs = float(decay10_epochs)
        start_lr = float(start_lr)
        peak_lr = float(peak_lr)

        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if reference_epochs <= 0 or warmup_epochs <= 0:
            raise ValueError("reference_epochs and warmup_epochs must be positive")
        if plateau_epochs < 0 or decay10_epochs <= 0:
            raise ValueError("plateau_epochs must be non-negative and decay10_epochs positive")
        if start_lr <= 0 or peak_lr <= 0 or start_lr > peak_lr:
            raise ValueError("learning rates must satisfy 0 < start_lr <= peak_lr")

        start_factor = start_lr / peak_lr
        plateau_end = warmup_epochs + plateau_epochs

        def standalone_factor(step):
            equivalent_epoch = (
                min(max(float(step), 0.0), float(total_steps))
                / float(total_steps)
                * reference_epochs
            )
            if equivalent_epoch <= warmup_epochs:
                progress = equivalent_epoch / warmup_epochs
                return start_factor * (1.0 / start_factor) ** progress
            if equivalent_epoch <= plateau_end:
                return 1.0
            decay_epochs = equivalent_epoch - plateau_end
            return 10.0 ** (-decay_epochs / decay10_epochs)

        super().__init__(
            optimizer=optimizer,
            lr_lambda=standalone_factor,
            last_epoch=last_epoch,
        )


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        start_value=0,
        warmup_iters=0,
        freeze_value=None,
        freeze_iters=0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)

        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = np.ones(freeze_iters) * freeze_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, schedule, freeze_schedule))
        self.iter = 0

    def get(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

    def step(self):
        value = self.get(self.iter)
        self.iter += 1
        return value

    def reset(self):
        self.iter = 0

    def __getitem__(self, it):
        return self.get(it)


def build_scheduler(cfg, optimizer):
    cfg = copy.deepcopy(cfg)
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
