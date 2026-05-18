#include "knn_query_cuda_kernel.h"
#include "../cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void knn_query_kernel(
    int m,
    int nsample,
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
    while (q_idx >= new_offset[batch_id]) {
        ++batch_id;
    }

    const int s_start = batch_id == 0 ? 0 : offset[batch_id - 1];
    const int s_end = offset[batch_id];
    const float *q_ptr = new_xyz + q_idx * 3;

    for (int k = 0; k < nsample; ++k) {
        idx[q_idx * nsample + k] = s_start;
        dist2[q_idx * nsample + k] = 1e30f;
    }

    for (int s_idx = s_start; s_idx < s_end; ++s_idx) {
        const float dx = q_ptr[0] - xyz[s_idx * 3 + 0];
        const float dy = q_ptr[1] - xyz[s_idx * 3 + 1];
        const float dz = q_ptr[2] - xyz[s_idx * 3 + 2];
        const float d2 = dx * dx + dy * dy + dz * dz;

        int insert_at = nsample;
        for (int k = 0; k < nsample; ++k) {
            if (d2 < dist2[q_idx * nsample + k]) {
                insert_at = k;
                break;
            }
        }

        if (insert_at < nsample) {
            for (int k = nsample - 1; k > insert_at; --k) {
                idx[q_idx * nsample + k] = idx[q_idx * nsample + k - 1];
                dist2[q_idx * nsample + k] = dist2[q_idx * nsample + k - 1];
            }
            idx[q_idx * nsample + insert_at] = s_idx;
            dist2[q_idx * nsample + insert_at] = d2;
        }
    }
}

}  // namespace

void knn_query_cuda_launcher(int m, int nsample,
                             const float *xyz, const float *new_xyz,
                             const int *offset, const int *new_offset,
                             int *idx, float *dist2) {
    const int threads = 256;
    const int blocks = DIVUP(m, threads);
    knn_query_kernel<<<blocks, threads>>>(
        m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2
    );
}
