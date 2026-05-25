import torch
import torch.nn as nn


class FeatureDropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class CoordPositionalEncoding(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim // 4)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, coord):
        if coord.numel() == 0:
            return coord.new_zeros((0, self.net[-1].out_features))

        coord = coord - coord.mean(dim=0, keepdim=True)
        scale = coord.abs().amax(dim=0, keepdim=True).clamp(min=1e-6)
        coord = coord / scale
        return self.net(coord)


class SGCAAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [1, L, C]
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            need_weights=False,
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SGCALite(nn.Module):
    """
    Stage-1 sparse global context adapter.

    Design goal:
        - self-contained, no PTv3 dependency
        - only used on low-resolution stages
        - compatible with KPConvXBase feature flow

    Input:
        feats  : [N, C]
        coord  : [N, 3]
        lengths: [B]

    Output:
        feats  : [N, C]
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        patch_size=256,
        context_ratio=1.0,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.context_ratio = float(context_ratio)

        context_dim = max(1, int(round(dim * self.context_ratio)))
        context_dim = min(dim, context_dim)
        num_heads = max(1, min(int(num_heads), context_dim))
        while context_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.context_dim = context_dim
        self.num_heads = num_heads

        self.in_proj = nn.Identity() if context_dim == dim else nn.Linear(dim, context_dim)
        self.out_proj = nn.Identity() if context_dim == dim else nn.Linear(context_dim, dim)
        self.pos_embed = CoordPositionalEncoding(context_dim)
        self.pre_norm = nn.LayerNorm(context_dim)
        self.block = SGCAAttentionBlock(
            dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.drop_path = FeatureDropPath(drop_path)

        self.global_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _serialization_order(coord):
        """
        Lightweight serialization proxy.
        We avoid PTv3 hard dependency here and build a stable ordering
        from normalized xyz coordinates.
        """
        coord = coord - coord.min(dim=0, keepdim=True)[0]
        scale = coord.max(dim=0, keepdim=True)[0].clamp(min=1e-6)
        coord = coord / scale

        key = coord[:, 0] + 2.17 * coord[:, 1] + 3.31 * coord[:, 2]
        order = torch.argsort(key)
        return order

    def _context_single_cloud(self, feats, coord):
        n = feats.shape[0]
        if n == 0:
            return feats, feats.new_zeros((1, self.dim))
        if n <= 1:
            return feats, feats.mean(dim=0, keepdim=True)

        x = self.in_proj(feats) + self.pos_embed(coord)
        x = self.pre_norm(x)

        order = self._serialization_order(coord)
        x_sorted = x[order]

        x_out_sorted = torch.empty_like(x_sorted)

        for start in range(0, n, self.patch_size):
            end = min(start + self.patch_size, n)
            patch = x_sorted[start:end].unsqueeze(0)  # [1, L, C]
            patch = self.block(patch).squeeze(0)
            x_out_sorted[start:end] = patch

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=order.device)
        x_out = x_out_sorted[inverse]

        x_context = self.out_proj(x_out)
        global_token = x_context.mean(dim=0, keepdim=True)
        return x_context, global_token

    def _forward_single_cloud(self, feats, coord):
        x_context, global_token = self._context_single_cloud(feats, coord)
        global_token = global_token.expand(feats.shape[0], -1)
        gate = self.global_gate(torch.cat([feats, global_token], dim=-1))
        fused = self.fuse(torch.cat([x_context, global_token], dim=-1))

        return feats + self.drop_path(gate * fused)

    def context(self, feats, coord, lengths):
        """
        Return one context token per cloud without modifying backbone features.
        This lets callers use global context as a decoder/head condition instead
        of injecting it into encoder skip features.
        """
        if feats.numel() == 0:
            return feats.new_zeros((0, self.dim))

        tokens = []
        start = 0

        for length in lengths.tolist():
            length = int(length)
            end = start + length

            feats_b = feats[start:end]
            coord_b = coord[start:end]

            if length == 0:
                token = feats.new_zeros((1, self.dim))
            else:
                _, token = self._context_single_cloud(feats_b, coord_b)
            tokens.append(token)
            start = end

        return torch.cat(tokens, dim=0)

    def forward(self, feats, coord, lengths):
        """
        feats  : [N, C]
        coord  : [N, 3]
        lengths: [B]
        """
        if feats.numel() == 0:
            return feats

        outputs = torch.empty_like(feats)
        start = 0

        for length in lengths.tolist():
            length = int(length)
            end = start + length

            feats_b = feats[start:end]
            coord_b = coord[start:end]

            outputs[start:end] = self._forward_single_cloud(feats_b, coord_b)
            start = end

        return outputs
