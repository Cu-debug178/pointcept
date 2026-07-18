import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.kpconvx_base import KPConvXBase
from pointcept.models.kpconvx.utils.kpnext_blocks import KPConvX

from .soka_lite_blocks import SOKALiteKPConvX


@MODELS.register_module("kpconvx_soka_lite")
class KPConvXSOKALite(KPConvXBase):
    """Plain KPConvX with SOKA-Lite in selected encoder attention stages."""

    def __init__(
        self,
        *args,
        soka_enabled=True,
        soka_stages=(2, 3, 4, 5),
        soka_hidden_dim=16,
        soka_bias_bound=2.0,
        soka_monitor=True,
        soka_eps=1.0e-6,
        **kwargs,
    ):
        self.soka_enabled = bool(soka_enabled)
        self.soka_stages = tuple(dict.fromkeys(int(stage) for stage in soka_stages))
        self.soka_hidden_dim = int(soka_hidden_dim)
        self.soka_bias_bound = float(soka_bias_bound)
        self.soka_monitor = bool(soka_monitor)
        self.soka_eps = float(soka_eps)
        super().__init__(*args, **kwargs)

        for stage in self.soka_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid SOKA stage index: {stage}")

        self._soka_modules_by_stage = {}
        if self.soka_enabled:
            self._install_soka_modules()

    def _install_soka_modules(self):
        for stage in self.soka_stages:
            stage_modules = []
            for block in getattr(self, f"encoder_{stage}"):
                conv = getattr(block, "conv", None)
                if not isinstance(conv, KPConvX):
                    continue
                if not isinstance(conv, SOKALiteKPConvX):
                    conv = SOKALiteKPConvX.from_kpconvx(
                        conv,
                        soka_hidden_dim=self.soka_hidden_dim,
                        soka_bias_bound=self.soka_bias_bound,
                        soka_monitor=self.soka_monitor,
                        soka_eps=self.soka_eps,
                    )
                    block.conv = conv
                stage_modules.append(conv)
            self._soka_modules_by_stage[int(stage)] = stage_modules

    def _collect_soka_monitor(self, ref_tensor):
        metrics = {}
        for stage in self.soka_stages:
            diag_by_key = {}
            for module in self._soka_modules_by_stage.get(int(stage), ()):
                for key, value in module._last_soka_diag.items():
                    diag_by_key.setdefault(key, []).append(value)
            for key, values in diag_by_key.items():
                metrics[f"soka_s{stage}_{key}"] = torch.stack(values).mean().to(
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
        return metrics

    def forward(self, data_dict):
        logits = super().forward(data_dict)
        if (
            self.soka_enabled
            and self.soka_monitor
            and self.task == "cloud_segmentation"
        ):
            output = {"seg_logits": logits}
            output.update(self._collect_soka_monitor(logits))
            return output
        return logits
