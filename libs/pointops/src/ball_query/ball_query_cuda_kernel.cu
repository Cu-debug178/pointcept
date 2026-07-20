#include "ball_query_cuda_kernel.h"
#include "../cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace {

__global__ void ball_query_kernel(
    int m,
    int nsample,
    int batch_size,
    float min_radius,
    float max_radius,
    const float *xyz,
    const float *new_xyz,
    const int *offset,
    const int *new_offset,
    int *idx,
    float *dist2) {
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= m) {
        return;
    }

    int batch_id = 0;
    while (batch_id < batch_size && q_idx >= new_offset[batch_id]) {
        ++batch_id;
    }

    if (batch_id >= batch_size) {
        return;
    }

    const int s_start = batch_id == 0 ? 0 : offset[batch_id - 1];
    const int s_end = offset[batch_id];
    const float min_radius2 = min_radius * min_radius;
    const float max_radius2 = max_radius * max_radius;
    const float *q_ptr = new_xyz + q_idx * 3;

    int write_count = 0;
    float best_dist = 1e30f;
    int best_idx = s_start;

    for (int s_idx = s_start; s_idx < s_end; ++s_idx) {
        const float dx = q_ptr[0] - xyz[s_idx * 3 + 0];
        const float dy = q_ptr[1] - xyz[s_idx * 3 + 1];
        const float dz = q_ptr[2] - xyz[s_idx * 3 + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;

        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = s_idx;
        }

        if (d2 >= min_radius2 && d2 <= max_radius2 && write_count < nsample) {
            idx[q_idx * nsample + write_count] = s_idx;
            dist2[q_idx * nsample + write_count] = d2;
            ++write_count;
            if (write_count == nsample) {
                break;
            }
        }
    }

    if (write_count == 0) {
        idx[q_idx * nsample] = best_idx;
        dist2[q_idx * nsample] = best_dist;
        write_count = 1;
    }

    for (int k = write_count; k < nsample; ++k) {
        idx[q_idx * nsample + k] = idx[q_idx * nsample + write_count - 1];
        dist2[q_idx * nsample + k] = dist2[q_idx * nsample + write_count - 1];
    }
}

__global__ void adaptive_ball_query_kernel(
    int m,
    int nsample,
    int batch_size,
    float min_radius,
    const float *radius,
    const float *xyz,
    const float *new_xyz,
    const int *offset,
    const int *new_offset,
    int *idx,
    float *dist2) {
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= m) {
        return;
    }

    int batch_id = 0;
    while (batch_id < batch_size && q_idx >= new_offset[batch_id]) {
        ++batch_id;
    }

    if (batch_id >= batch_size) {
        return;
    }

    const int s_start = batch_id == 0 ? 0 : offset[batch_id - 1];
    const int s_end = offset[batch_id];
    const float radius_q = fmaxf(radius[q_idx], min_radius);
    const float min_radius2 = min_radius * min_radius;
    const float radius2 = radius_q * radius_q;
    const float *q_ptr = new_xyz + q_idx * 3;

    int write_count = 0;

    for (int s_idx = s_start; s_idx < s_end; ++s_idx) {
        const float dx = q_ptr[0] - xyz[s_idx * 3 + 0];
        const float dy = q_ptr[1] - xyz[s_idx * 3 + 1];
        const float dz = q_ptr[2] - xyz[s_idx * 3 + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;

        if (d2 >= min_radius2 && d2 <= radius2 && write_count < nsample) {
            idx[q_idx * nsample + write_count] = s_idx;
            if (dist2 != nullptr) {
                dist2[q_idx * nsample + write_count] = d2;
            }
            ++write_count;
            if (write_count == nsample) {
                break;
            }
        }
    }

    for (int k = write_count; k < nsample; ++k) {
        idx[q_idx * nsample + k] = -1;
        if (dist2 != nullptr) {
            dist2[q_idx * nsample + k] = 0.0f;
        }
    }
}

}  // namespace

void ball_query_cuda_launcher(int m, int nsample, int batch_size,
                              float min_radius, float max_radius,
                              const float *xyz, const float *new_xyz,
                              const int *offset, const int *new_offset,
                              int *idx, float *dist2, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = DIVUP(m, threads);
    ball_query_kernel<<<blocks, threads, 0, stream>>>(
        m, nsample, batch_size, min_radius, max_radius, xyz, new_xyz,
        offset, new_offset, idx, dist2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaptive_ball_query_cuda_launcher(int m, int nsample, int batch_size,
                                       float min_radius,
                                       const float *radius,
                                       const float *xyz, const float *new_xyz,
                                       const int *offset, const int *new_offset,
                                       int *idx, float *dist2, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = DIVUP(m, threads);
    adaptive_ball_query_kernel<<<blocks, threads, 0, stream>>>(
        m, nsample, batch_size, min_radius, radius, xyz, new_xyz,
        offset, new_offset, idx, dist2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adaptive_ball_query_idx_cuda_launcher(int m, int nsample, int batch_size,
                                           float min_radius,
                                           const float *radius,
                                           const float *xyz, const float *new_xyz,
                                           const int *offset, const int *new_offset,
                                           int *idx, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = DIVUP(m, threads);
    adaptive_ball_query_kernel<<<blocks, threads, 0, stream>>>(
        m, nsample, batch_size, min_radius, radius, xyz, new_xyz,
        offset, new_offset, idx, nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
