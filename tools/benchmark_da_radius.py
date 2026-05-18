import argparse
import time

import torch


def make_points(num_points, batches, device):
    assert num_points % batches == 0
    per_batch = num_points // batches
    points = []
    for batch_id in range(batches):
        pts = torch.rand(per_batch, 3, device=device)
        pts = pts * (1.0 + 0.15 * batch_id) + batch_id * 2.0
        points.append(pts)
    points = torch.cat(points, dim=0).contiguous()
    offset = torch.arange(
        per_batch,
        num_points + 1,
        per_batch,
        dtype=torch.int32,
        device=device,
    )
    return points, offset


def offset_to_lengths(offset):
    starts = torch.cat([offset.new_zeros(1), offset[:-1]])
    return (offset - starts).long()


def density_radius_scale(points, offset, k, scale_range=(0.8, 1.5), strength=0.75):
    lengths = offset_to_lengths(offset)
    scales = []
    start = 0
    eps = 1e-6

    for length in lengths.tolist():
        end = start + int(length)
        pts = points[start:end]
        dist = torch.cdist(pts, pts)
        kk = min(k + 1, pts.shape[0])
        knn_dist = torch.topk(dist, k=kk, largest=False).values[:, 1:]
        mean_dist = knn_dist.mean(dim=1, keepdim=True)
        density = 1.0 / (mean_dist + eps)
        lo, hi = torch.quantile(density.flatten(), torch.tensor([0.1, 0.9], device=points.device))
        density_norm = ((density - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
        sparse_score = 1.0 - density_norm
        raw_scale = scale_range[0] + (scale_range[1] - scale_range[0]) * sparse_score
        scale = 1.0 + strength * (raw_scale - 1.0)
        scales.append(scale.clamp(min=scale_range[0], max=scale_range[1]).flatten())
        start = end

    return torch.cat(scales, dim=0)


def torch_radius_mask(points, offset, radius, nsample):
    rows = []
    distances = []
    start = 0

    for end in offset.tolist():
        pts = points[start:end]
        dist = torch.cdist(pts, pts)
        valid = dist <= radius[start:end].view(-1, 1)
        masked_dist = dist.masked_fill(~valid, float("inf"))
        kk = min(nsample, pts.shape[0])
        vals, idx = torch.topk(masked_dist, k=kk, largest=False)

        empty = torch.isinf(vals[:, 0])
        if empty.any():
            fallback_vals, fallback_idx = torch.topk(dist[empty], k=1, largest=False)
            vals[empty, 0:1] = fallback_vals
            idx[empty, 0:1] = fallback_idx

        finite = torch.isfinite(vals)
        for row_idx in range(vals.shape[0]):
            invalid = ~finite[row_idx]
            if invalid.any():
                valid_pos = torch.nonzero(finite[row_idx], as_tuple=False).flatten()
                if valid_pos.numel() == 0:
                    nearest_val, nearest_idx = torch.topk(dist[row_idx], k=1, largest=False)
                    vals[row_idx, 0] = nearest_val[0]
                    idx[row_idx, 0] = nearest_idx[0]
                    fill_pos = 0
                else:
                    fill_pos = valid_pos[-1].item()
                vals[row_idx, invalid] = vals[row_idx, fill_pos].clone()
                idx[row_idx, invalid] = idx[row_idx, fill_pos].clone()

        if kk < nsample:
            idx = torch.cat([idx, idx[:, -1:].expand(-1, nsample - kk)], dim=1)
            vals = torch.cat([vals, vals[:, -1:].expand(-1, nsample - kk)], dim=1)

        idx = idx + start
        rows.append(idx.int())
        distances.append(vals)
        start = end

    return torch.cat(rows, dim=0), torch.cat(distances, dim=0)


def unique_neighbor_count(idx):
    return torch.tensor(
        [torch.unique(row).numel() for row in idx],
        dtype=torch.float32,
        device=idx.device,
    )


def time_cuda(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeat
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    return elapsed_ms, peak_mb, out


def time_cpu(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeat
    return elapsed_ms, 0.0, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--nsample", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--density-k", type=int, default=16)
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--scale-strength", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("pointops benchmark currently requires CUDA")

    import pointops

    if not hasattr(pointops, "adaptive_ball_query"):
        raise RuntimeError("pointops.adaptive_ball_query is unavailable")

    points, offset = make_points(args.num_points, args.batches, device)
    radius_scale = density_radius_scale(
        points,
        offset,
        k=args.density_k,
        scale_range=(args.scale_min, args.scale_max),
        strength=args.scale_strength,
    )
    radius = args.radius * radius_scale

    timer = time_cuda if device.type == "cuda" else time_cpu
    adaptive_ms, adaptive_peak_mb, (idx, dist) = timer(
        lambda: pointops.adaptive_ball_query(
            args.nsample,
            0.0,
            radius,
            points,
            offset,
        ),
        args.warmup,
        args.repeat,
    )
    outside_count = (dist > radius.view(-1, 1) + 1e-4).sum().item()
    adaptive_unique_count = unique_neighbor_count(idx)

    knn_ms, knn_peak_mb, (knn_idx, _) = timer(
        lambda: pointops.knn_query(args.nsample, points, offset),
        args.warmup,
        args.repeat,
    )
    knn_unique_count = unique_neighbor_count(knn_idx)

    torch_ms, torch_peak_mb, (torch_idx, torch_dist) = timer(
        lambda: torch_radius_mask(points, offset, radius, args.nsample),
        args.warmup,
        args.repeat,
    )
    torch_outside_count = (torch_dist > radius.view(-1, 1) + 1e-4).sum().item()
    torch_unique_count = unique_neighbor_count(torch_idx)

    print("da_radius_benchmark:")
    print(f"  device={device}")
    print(f"  points={args.num_points}")
    print(f"  batches={args.batches}")
    print(f"  nsample={args.nsample}")
    print(f"  radius_base={args.radius:.4f}")
    print(f"  radius_scale_min={radius_scale.min().item():.4f}")
    print(f"  radius_scale_mean={radius_scale.mean().item():.4f}")
    print(f"  radius_scale_max={radius_scale.max().item():.4f}")
    print(f"  radius_mean={radius.mean().item():.4f}")
    print(f"  adaptive_ball_query_ms={adaptive_ms:.3f}")
    print(f"  adaptive_peak_mb={adaptive_peak_mb:.2f}")
    print(f"  adaptive_unique_neighbors_min={adaptive_unique_count.min().item():.2f}")
    print(f"  adaptive_unique_neighbors_mean={adaptive_unique_count.mean().item():.2f}")
    print(f"  adaptive_unique_neighbors_max={adaptive_unique_count.max().item():.2f}")
    print(f"  adaptive_outside_count={outside_count}")
    print(f"  torch_mask_ms={torch_ms:.3f}")
    print(f"  torch_mask_peak_mb={torch_peak_mb:.2f}")
    print(f"  torch_mask_unique_neighbors_mean={torch_unique_count.mean().item():.2f}")
    print(f"  torch_mask_outside_count={torch_outside_count}")
    print(f"  knn_query_ms={knn_ms:.3f}")
    print(f"  knn_peak_mb={knn_peak_mb:.2f}")
    print(f"  knn_unique_neighbors_mean={knn_unique_count.mean().item():.2f}")


if __name__ == "__main__":
    main()
