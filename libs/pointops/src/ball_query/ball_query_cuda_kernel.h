#ifndef _BALL_QUERY_CUDA_KERNEL
#define _BALL_QUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void ball_query_cuda(int m, int nsample,
                     float min_radius, float max_radius,
                     at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                     at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                     at::Tensor idx_tensor, at::Tensor dist2_tensor);

void adaptive_ball_query_cuda(int m, int nsample,
                              float min_radius,
                              at::Tensor radius_tensor,
                              at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                              at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                              at::Tensor idx_tensor, at::Tensor dist2_tensor);

void adaptive_ball_query_idx_cuda(int m, int nsample,
                                  float min_radius,
                                  at::Tensor radius_tensor,
                                  at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                                  at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                                  at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void ball_query_cuda_launcher(int m, int nsample, int batch_size,
                              float min_radius, float max_radius,
                              const float *xyz, const float *new_xyz,
                              const int *offset, const int *new_offset,
                              int *idx, float *dist2, cudaStream_t stream);

void adaptive_ball_query_cuda_launcher(int m, int nsample, int batch_size,
                                       float min_radius,
                                       const float *radius,
                                       const float *xyz, const float *new_xyz,
                                       const int *offset, const int *new_offset,
                                       int *idx, float *dist2, cudaStream_t stream);

void adaptive_ball_query_idx_cuda_launcher(int m, int nsample, int batch_size,
                                           float min_radius,
                                           const float *radius,
                                           const float *xyz, const float *new_xyz,
                                           const int *offset, const int *new_offset,
                                           int *idx, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
