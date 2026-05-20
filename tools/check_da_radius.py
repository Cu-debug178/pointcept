import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBS_ROOT = REPO_ROOT / "libs"
if (LIBS_ROOT / "pointops").exists() and str(LIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBS_ROOT))


def knn_candidates(points, offset, k):
    rows = []
    start = 0
    for end in offset.tolist():
        pts = points[start:end]
        dist = torch.cdist(pts, pts)
        kk = min(k, pts.shape[0])
        idx = torch.topk(dist, k=kk, largest=False).indices + start
        if kk < k:
            pad = idx[:, -1:].expand(-1, k - kk)
            idx = torch.cat([idx, pad], dim=1)
        rows.append(idx)
        start = end
    return torch.cat(rows, dim=0)


def density_adaptive_radius(points, neighbors, lengths, scale_range=(0.8, 1.5)):
    eps = 1e-6
    safe_neighbors = neighbors.clamp(min=0, max=points.shape[0] - 1)
    valid = ((neighbors >= 0) & (neighbors < points.shape[0])).float()
    dist = torch.norm(points[safe_neighbors] - points.unsqueeze(1), dim=-1)
    valid = valid * (dist > eps).float()
    valid_count = valid.sum(dim=1, keepdim=True)
    mean_dist = (dist * valid).sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)
    no_neighbor = valid_count <= 0
    if no_neighbor.any():
        fallback = mean_dist[~no_neighbor].mean() if (~no_neighbor).any() else points.new_tensor(1.0)
        mean_dist = torch.where(no_neighbor, fallback.expand_as(mean_dist), mean_dist)
    rho = 1.0 / (mean_dist + eps)

    out = torch.empty_like(rho)
    start = 0
    for length in lengths.tolist():
        end = start + int(length)
        rho_b = rho[start:end]
        q = torch.tensor((0.1, 0.9), dtype=rho_b.dtype, device=rho_b.device)
        lo, hi = torch.quantile(rho_b.flatten(), q)
        rho_norm = ((rho_b - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
        sparse_score = 1.0 - rho_norm
        raw_scale = scale_range[0] + (scale_range[1] - scale_range[0]) * sparse_score
        out[start:end] = 1.0 + 0.75 * (raw_scale - 1.0)
        start = end
    return out.clamp(min=scale_range[0], max=scale_range[1])


def check_torch_reference(device):
    torch.manual_seed(7)
    points = torch.rand(96, 3, device=device)
    points[48:] = points[48:] * 2.5 + 2.0
    offset = torch.tensor([48, 96], dtype=torch.int32, device=device)
    lengths = torch.tensor([48, 48], dtype=torch.long, device=device)
    neighbors = knn_candidates(points, offset, 16)

    radius_scale = density_adaptive_radius(points, neighbors, lengths)
    base_radius = 0.2
    safe_neighbors = neighbors.clamp(min=0, max=points.shape[0] - 1)
    dist = torch.norm(points[safe_neighbors] - points.unsqueeze(1), dim=-1)
    valid = dist <= base_radius * radius_scale

    print("torch_reference:")
    print(f"  points={points.shape[0]} k={neighbors.shape[1]}")
    print(f"  radius_scale_min={radius_scale.min().item():.4f}")
    print(f"  radius_scale_mean={radius_scale.mean().item():.4f}")
    print(f"  radius_scale_max={radius_scale.max().item():.4f}")
    print(f"  valid_ratio={valid.float().mean().item():.4f}")


def check_cuda_query():
    if not torch.cuda.is_available():
        print("cuda_query: skipped, torch.cuda.is_available() is False")
        return

    try:
        import pointops
    except Exception as exc:
        print(f"cuda_query: skipped, import pointops failed: {exc}")
        return

    if not hasattr(pointops, "adaptive_ball_query"):
        print("cuda_query: skipped, pointops.adaptive_ball_query is unavailable")
        return

    torch.manual_seed(11)
    points = torch.rand(128, 3, device="cuda").contiguous()
    points[64:] = points[64:] * 1.75 + 1.25
    offset = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
    radius = torch.linspace(0.16, 0.32, points.shape[0], device="cuda")
    idx, dist = pointops.adaptive_ball_query(16, 0.0, radius, points, offset)
    torch.cuda.synchronize()
    valid_idx = idx >= 0
    max_over = ((dist > radius.view(-1, 1) + 1e-4) & valid_idx).sum().item()
    shadow_ratio = (~valid_idx).float().mean().item()

    brute_counts = []
    cuda_unique_counts = []
    start = 0
    for end in offset.tolist():
        pts = points[start:end]
        pair_dist = torch.cdist(pts, pts)
        local_radius = radius[start:end].view(-1, 1)
        brute_counts.append((pair_dist <= local_radius + 1e-4).sum(dim=1).clamp(max=16))
        local_idx = idx[start:end] - start
        cuda_unique_counts.append(
            torch.tensor(
                [torch.unique(row[row >= 0]).numel() for row in local_idx],
                dtype=torch.float32,
                device=points.device,
            )
        )
        start = end
    brute_counts = torch.cat(brute_counts).float()
    cuda_unique_counts = torch.cat(cuda_unique_counts).float()

    bench_points = torch.rand(1024, 3, device="cuda").contiguous()
    bench_offset = torch.tensor([1024], dtype=torch.int32, device="cuda")
    bench_radius = torch.full((1024,), 0.08, dtype=torch.float32, device="cuda")
    repeats = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(repeats):
        pointops.adaptive_ball_query(16, 0.0, bench_radius, bench_points, bench_offset)
    end_event.record()
    torch.cuda.synchronize()
    adaptive_ms = start_event.elapsed_time(end_event) / repeats

    start_event.record()
    for _ in range(repeats):
        pointops.knn_query(16, bench_points, bench_offset)
    end_event.record()
    torch.cuda.synchronize()
    knn_ms = start_event.elapsed_time(end_event) / repeats

    print("cuda_query:")
    print(f"  idx_shape={tuple(idx.shape)}")
    print(f"  max_distance={dist.max().item():.4f}")
    print(f"  outside_count={max_over}")
    print(f"  shadow_ratio={shadow_ratio:.4f}")
    print(f"  brute_count_mean={brute_counts.mean().item():.2f}")
    print(f"  cuda_unique_count_mean={cuda_unique_counts.mean().item():.2f}")
    print(f"  adaptive_ball_query_ms={adaptive_ms:.3f}")
    print(f"  knn_query_ms={knn_ms:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    check_torch_reference(device)
    if args.device == "cuda":
        check_cuda_query()


if __name__ == "__main__":
    main()
