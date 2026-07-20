#include <vector>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "ball_query_cuda_kernel.h"

namespace {

void check_query_inputs(
    int m,
    int nsample,
    const at::Tensor &xyz,
    const at::Tensor &new_xyz,
    const at::Tensor &offset,
    const at::Tensor &new_offset,
    const at::Tensor &idx,
    const at::Tensor *dist2,
    const at::Tensor *radius) {
    TORCH_CHECK(m >= 0, "m must be non-negative");
    TORCH_CHECK(nsample > 0, "nsample must be positive");
    TORCH_CHECK(xyz.is_cuda() && new_xyz.is_cuda(), "xyz and new_xyz must be CUDA tensors");
    TORCH_CHECK(offset.is_cuda() && new_offset.is_cuda(), "offset tensors must be CUDA tensors");
    TORCH_CHECK(xyz.scalar_type() == at::kFloat && new_xyz.scalar_type() == at::kFloat,
                "xyz and new_xyz must be float32");
    TORCH_CHECK(offset.scalar_type() == at::kInt && new_offset.scalar_type() == at::kInt,
                "offset tensors must be int32");
    TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape [N, 3]");
    TORCH_CHECK(new_xyz.dim() == 2 && new_xyz.size(1) == 3,
                "new_xyz must have shape [M, 3]");
    TORCH_CHECK(m == new_xyz.size(0), "m does not match new_xyz.size(0)");
    TORCH_CHECK(offset.dim() == 1 && new_offset.dim() == 1,
                "offset tensors must be one-dimensional");
    TORCH_CHECK(offset.numel() > 0 && offset.numel() == new_offset.numel(),
                "offset and new_offset must have the same non-empty batch size");
    TORCH_CHECK(xyz.device() == new_xyz.device() && xyz.device() == offset.device() &&
                    xyz.device() == new_offset.device() && xyz.device() == idx.device(),
                "all query tensors must be on the same CUDA device");
    TORCH_CHECK(xyz.is_contiguous() && new_xyz.is_contiguous() && offset.is_contiguous() &&
                    new_offset.is_contiguous() && idx.is_contiguous(),
                "query tensors must be contiguous");
    TORCH_CHECK(idx.dim() == 2 && idx.size(0) == m && idx.size(1) == nsample,
                "idx must have shape [m, nsample]");
    if (dist2 != nullptr) {
        TORCH_CHECK(dist2->scalar_type() == at::kFloat && dist2->is_cuda() &&
                        dist2->device() == xyz.device() && dist2->is_contiguous(),
                    "dist2 must be contiguous CUDA float32 on the same device");
        TORCH_CHECK(dist2->dim() == 2 && dist2->size(0) == m && dist2->size(1) == nsample,
                    "dist2 must have shape [m, nsample]");
    }
    if (radius != nullptr) {
        TORCH_CHECK(radius->is_cuda() && radius->device() == xyz.device() &&
                        radius->scalar_type() == at::kFloat && radius->is_contiguous(),
                    "radius must be contiguous CUDA float32 on the same device");
        TORCH_CHECK(radius->numel() == m, "radius must contain one value per query point");
    }
}

}  // namespace


void ball_query_cuda(int m, int nsample,
                     float min_radius, float max_radius,
                     at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                     at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                     at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    check_query_inputs(m, nsample, xyz_tensor, new_xyz_tensor, offset_tensor,
                       new_offset_tensor, idx_tensor, &dist2_tensor, nullptr);
    if (m == 0) {
        return;
    }
    const c10::cuda::CUDAGuard device_guard(xyz_tensor.device());
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    const int batch_size = static_cast<int>(offset_tensor.numel());
    ball_query_cuda_launcher(
        m, nsample, batch_size, min_radius, max_radius, xyz, new_xyz,
        offset, new_offset, idx, dist2,
        at::cuda::getCurrentCUDAStream());
}

void adaptive_ball_query_cuda(int m, int nsample,
                              float min_radius,
                              at::Tensor radius_tensor,
                              at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                              at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                              at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    check_query_inputs(m, nsample, xyz_tensor, new_xyz_tensor, offset_tensor,
                       new_offset_tensor, idx_tensor, &dist2_tensor, &radius_tensor);
    if (m == 0) {
        return;
    }
    const c10::cuda::CUDAGuard device_guard(xyz_tensor.device());
    const float *radius = radius_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    const int batch_size = static_cast<int>(offset_tensor.numel());
    adaptive_ball_query_cuda_launcher(
        m, nsample, batch_size, min_radius, radius, xyz, new_xyz,
        offset, new_offset, idx, dist2,
        at::cuda::getCurrentCUDAStream());
}

void adaptive_ball_query_idx_cuda(int m, int nsample,
                                  float min_radius,
                                  at::Tensor radius_tensor,
                                  at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                                  at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                                  at::Tensor idx_tensor) {
    check_query_inputs(m, nsample, xyz_tensor, new_xyz_tensor, offset_tensor,
                       new_offset_tensor, idx_tensor, nullptr, &radius_tensor);
    if (m == 0) {
        return;
    }
    const c10::cuda::CUDAGuard device_guard(xyz_tensor.device());
    const float *radius = radius_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    const int batch_size = static_cast<int>(offset_tensor.numel());
    adaptive_ball_query_idx_cuda_launcher(
        m, nsample, batch_size, min_radius, radius, xyz, new_xyz,
        offset, new_offset, idx,
        at::cuda::getCurrentCUDAStream());
}
