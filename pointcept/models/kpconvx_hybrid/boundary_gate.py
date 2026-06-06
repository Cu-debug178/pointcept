import torch
import torch.nn as nn


class BoundaryRiskHead(nn.Module):
    """
    Lightweight point-level boundary risk predictor.

    It reuses finest-level decoder features and stage-0 neighbors. The module is
    intentionally local: its output is used to attenuate global residuals near
    class boundaries instead of replacing the semantic classifier.
    """

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim // 2)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid

    @staticmethod
    def masked_mean(values, valid_mask):
        weight = valid_mask.float().unsqueeze(-1)
        denom = weight.sum(dim=1).clamp(min=1.0)
        return (values * weight).sum(dim=1) / denom

    @staticmethod
    @torch.no_grad()
    def build_target(segment, neighbors, ignore_index=-1, dilate_steps=1):
        """
        Build online pseudo boundary labels after data augmentation.

        A point is a boundary point if any valid neighbor has a different class.
        Optional dilation turns razor-thin edges into a small boundary band.
        """
        num_points = int(segment.shape[0])
        if num_points == 0:
            target = segment.new_zeros((0,), dtype=torch.float32)
            valid_mask = segment.new_zeros((0,), dtype=torch.bool)
            return target, valid_mask

        safe_neighbors, valid = BoundaryRiskHead.sanitize_neighbors(
            neighbors, num_points
        )
        center = segment.reshape(-1)
        neighbor_label = center[safe_neighbors]
        center_valid = center != ignore_index
        neighbor_valid = valid & (neighbor_label != ignore_index)
        different = (
            center_valid.unsqueeze(1)
            & neighbor_valid
            & (neighbor_label != center.unsqueeze(1))
        )
        boundary = different.any(dim=1)

        for _ in range(max(int(dilate_steps), 0)):
            neighbor_boundary = boundary[safe_neighbors] & valid
            boundary = boundary | neighbor_boundary.any(dim=1)

        boundary = boundary & center_valid
        return boundary.float(), center_valid

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            return feats.new_zeros((0,))

        x = self.norm(feats)
        safe_neighbors, valid = self.sanitize_neighbors(neighbors, feats.shape[0])
        mean_feat = self.masked_mean(x[safe_neighbors], valid)
        mean_point = self.masked_mean(points[safe_neighbors], valid)

        feat_diff = torch.norm(x - mean_feat, dim=-1, keepdim=True)
        coord_diff = torch.norm(points - mean_point, dim=-1, keepdim=True)
        valid_ratio = valid.float().mean(dim=1, keepdim=True)
        head_input = torch.cat(
            [x, mean_feat, feat_diff, coord_diff, valid_ratio],
            dim=-1,
        )
        return self.mlp(head_input).squeeze(-1)
