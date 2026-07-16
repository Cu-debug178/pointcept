import torch
import torch.nn as nn
from torch import Tensor

from pointcept.models.kpconvx.utils.generic_blocks import index_select
from pointcept.models.kpconvx.utils.kpnext_blocks import KPConvX

from .soka_geometry import build_soka_descriptor


def _attention_name(module: KPConvX):
    if module.attention_act is torch.sigmoid:
        return "sigmoid"
    if module.attention_act is torch.tanh:
        return "tanh"
    if isinstance(module.attention_act, nn.Softmax):
        return "softmax"
    if isinstance(module.attention_act, nn.Identity):
        return "none"
    raise ValueError("Unsupported KPConvX attention activation")


class SOKAKPConvX(KPConvX):
    """KPConvX with a zero-initialized geometry bias on attention logits."""

    def __init__(
        self,
        *args,
        soka_enabled=True,
        soka_hidden_dim=16,
        soka_bias_bound=2.0,
        soka_monitor=False,
        soka_eps=1.0e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if int(soka_hidden_dim) < 1:
            raise ValueError("soka_hidden_dim must be positive")
        if float(soka_bias_bound) < 0:
            raise ValueError("soka_bias_bound must be non-negative")

        self.soka_enabled = bool(soka_enabled)
        self.soka_bias_bound = float(soka_bias_bound)
        self.soka_monitor = bool(soka_monitor)
        self.soka_eps = float(soka_eps)
        self.soka_mlp = nn.Sequential(
            nn.Linear(6, int(soka_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(soka_hidden_dim), 1),
        )
        nn.init.zeros_(self.soka_mlp[-1].weight)
        nn.init.zeros_(self.soka_mlp[-1].bias)
        self._last_soka_diag = {}

    @classmethod
    def from_kpconvx(
        cls,
        module: KPConvX,
        *,
        soka_hidden_dim=16,
        soka_bias_bound=2.0,
        soka_monitor=False,
        soka_eps=1.0e-6,
    ):
        upgraded = cls(
            module.channels,
            module.shell_sizes,
            module.radius,
            module.sigma,
            attention_groups=module.groups,
            attention_act=_attention_name(module),
            mod_grp_norm=module.mod_grp_norm,
            shared_kp_data=module.shared_kp_data if module.share_kp else None,
            dimension=module.dimension,
            influence_mode=module.influence_mode,
            fixed_kernel_points=module.fixed_kernel_points,
            inf=module.inf,
            soka_enabled=True,
            soka_hidden_dim=soka_hidden_dim,
            soka_bias_bound=soka_bias_bound,
            soka_monitor=soka_monitor,
            soka_eps=soka_eps,
        )

        # Preserve the exact baseline parameters, buffers, module behavior, and keys.
        upgraded.weights = module.weights
        upgraded.alpha_mlp = module.alpha_mlp
        upgraded.grpnorm = module.grpnorm
        upgraded.kernel_points = module.kernel_points
        upgraded.shared_kp_data = module.shared_kp_data
        upgraded.share_kp = module.share_kp
        upgraded.first_kp = module.first_kp
        upgraded.soka_mlp.to(
            device=module.weights.device,
            dtype=module.weights.dtype,
        )
        upgraded.train(module.training)
        return upgraded

    def _geometry_signature(self, q_pts: Tensor, s_pts: Tensor, neighb_inds: Tensor):
        return (
            q_pts.device.type,
            q_pts.device.index,
            q_pts.data_ptr(),
            q_pts._version,
            tuple(q_pts.shape),
            tuple(q_pts.stride()),
            q_pts.storage_offset(),
            s_pts.data_ptr(),
            s_pts._version,
            tuple(s_pts.shape),
            tuple(s_pts.stride()),
            s_pts.storage_offset(),
            neighb_inds.data_ptr(),
            neighb_inds._version,
            tuple(neighb_inds.shape),
            tuple(neighb_inds.stride()),
            neighb_inds.storage_offset(),
            self.kernel_points.data_ptr(),
            self.kernel_points._version,
            tuple(self.kernel_points.shape),
        )

    @torch.no_grad()
    def get_neighbors_influences(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        neighb_inds: Tensor,
        return_geometry=False,
    ):
        signature = self._geometry_signature(q_pts, s_pts, neighb_inds)
        cache_matches = (
            self.share_kp
            and self.shared_kp_data.get("soka_geometry_signature") == signature
        )

        force_recompute = self.share_kp and not self.first_kp and not cache_matches
        if force_recompute:
            original_first_kp = self.first_kp
            self.first_kp = True
            try:
                influence_weights, neighbors, neighbors_1nn = (
                    super().get_neighbors_influences(q_pts, s_pts, neighb_inds)
                )
            finally:
                self.first_kp = original_first_kp
        else:
            influence_weights, neighbors, neighbors_1nn = (
                super().get_neighbors_influences(q_pts, s_pts, neighb_inds)
            )

        use_cached_geometry = (
            cache_matches
            and not self.first_kp
            and "soka_nn_sq_dists" in self.shared_kp_data
            and "soka_valid_mask" in self.shared_kp_data
        )
        if use_cached_geometry:
            nn_sq_dists = self.shared_kp_data["soka_nn_sq_dists"]
            valid_mask = self.shared_kp_data["soka_valid_mask"]
        else:
            valid_mask = neighb_inds < s_pts.shape[0]
            assigned_kernels = self.kernel_points.index_select(
                0, neighbors_1nn.reshape(-1)
            ).reshape(*neighbors.shape)
            safe_neighbors = torch.where(
                valid_mask.unsqueeze(-1), neighbors, assigned_kernels
            )
            nn_sq_dists = (safe_neighbors - assigned_kernels).square().sum(dim=-1)
            nn_sq_dists = nn_sq_dists.masked_fill(~valid_mask, 0.0)
            if self.share_kp:
                self.shared_kp_data["soka_geometry_signature"] = signature
                self.shared_kp_data["soka_nn_sq_dists"] = nn_sq_dists
                self.shared_kp_data["soka_valid_mask"] = valid_mask

        if return_geometry:
            return (
                influence_weights,
                neighbors,
                neighbors_1nn,
                nn_sq_dists,
                valid_mask,
            )
        return influence_weights, neighbors, neighbors_1nn

    @staticmethod
    def _mean_std(values: Tensor, mask=None):
        if mask is not None:
            values = values[mask]
        values = values.reshape(-1)
        if values.numel() == 0:
            zero = values.new_zeros(())
            return zero, zero
        return values.mean(), values.std(unbiased=False)

    @torch.no_grad()
    def _record_soka_diag(
        self,
        descriptor,
        auxiliary,
        soka_bias,
        base_logits,
        attention,
    ):
        occupied = auxiliary["occupied_mask"]
        occ_mean, occ_std = self._mean_std(descriptor[..., 0])
        entropy_mean, entropy_std = self._mean_std(auxiliary["entropy"])
        radial_mean, radial_std = self._mean_std(descriptor[..., 1], occupied)
        assignment_mean, assignment_std = self._mean_std(
            descriptor[..., 2], occupied
        )
        bias_mean, bias_std = self._mean_std(soka_bias)
        bias_rms = soka_bias.float().square().mean().sqrt()
        base_rms = base_logits.float().square().mean().sqrt()
        self._last_soka_diag = {
            "occ_mean": occ_mean.detach(),
            "occ_std": occ_std.detach(),
            "entropy_mean": entropy_mean.detach(),
            "entropy_std": entropy_std.detach(),
            "radial_mean": radial_mean.detach(),
            "radial_std": radial_std.detach(),
            "assignment_mean": assignment_mean.detach(),
            "assignment_std": assignment_std.detach(),
            "bias_mean": bias_mean.detach(),
            "bias_std": bias_std.detach(),
            "bias_abs": soka_bias.detach().abs().mean(),
            "base_logit_abs": base_logits.detach().abs().mean(),
            "base_logit_rms": base_rms.detach(),
            "soka_to_base_ratio": (
                bias_rms / (base_rms + self.soka_eps)
            ).detach(),
            "attention_low_ratio": (attention.detach() < 0.05).float().mean(),
            "attention_high_ratio": (attention.detach() > 0.95).float().mean(),
        }

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        s_feats: Tensor,
        neighb_inds: Tensor,
    ) -> Tensor:
        if not self.soka_enabled:
            return super().forward(q_pts, s_pts, s_feats, neighb_inds)

        padded_s_feats = torch.cat(
            (s_feats, torch.zeros_like(s_feats[:1, :])), dim=0
        )
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        if q_pts.shape[0] == s_pts.shape[0]:
            pooled_feats = s_feats
        else:
            pooled_feats = neighbor_feats[:, 0, :]

        (
            influence_weights,
            relative_neighbors,
            neighbors_1nn,
            nn_sq_dists,
            valid_mask,
        ) = self.get_neighbors_influences(
            q_pts,
            s_pts,
            neighb_inds,
            return_geometry=True,
        )
        descriptor, auxiliary = build_soka_descriptor(
            relative_neighbors,
            neighbors_1nn,
            nn_sq_dists,
            valid_mask,
            self.kernel_points,
            self.radius,
            self.sigma,
            eps=self.soka_eps,
        )

        modulations = self.alpha_mlp(pooled_feats)
        if self.mod_grp_norm:
            modulations = modulations.transpose(0, 1).unsqueeze(0)
            modulations = self.grpnorm(modulations)
            modulations = modulations.squeeze(0).transpose(0, 1)

        base_logits = modulations.reshape(-1, self.K, self.ch_per_grp)
        soka_bias = self.soka_bias_bound * torch.tanh(
            self.soka_mlp(descriptor.to(dtype=base_logits.dtype))
        )
        attention_logits = (base_logits + soka_bias).reshape(
            -1, self.K * self.ch_per_grp
        )
        attention = self.attention_act(attention_logits).reshape(
            -1, self.K, self.ch_per_grp
        )

        conv_weights = self.weights.reshape(
            1, self.K, self.ch_per_grp, self.groups
        )
        conv_weights = conv_weights * attention.unsqueeze(-1)
        conv_weights = conv_weights.reshape(-1, self.K, self.channels)
        neighbors_weights = torch.gather(
            conv_weights,
            1,
            neighbors_1nn.unsqueeze(2).expand(-1, -1, self.channels),
        )
        if self.influence_mode != "constant":
            neighbors_weights = neighbors_weights * influence_weights.unsqueeze(2)

        output_feats = self.aggr_op(
            self.merge_op(neighbor_feats, neighbors_weights), dim=1
        )
        if self.soka_monitor:
            self._record_soka_diag(
                descriptor,
                auxiliary,
                soka_bias,
                base_logits,
                attention,
            )
        return output_feats
