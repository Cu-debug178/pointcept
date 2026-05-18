#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void missing_pointops_kernel(const char *name) {
    throw std::runtime_error(
        std::string(name) +
        " is not available because this checkout does not include its CUDA kernel source."
    );
}

}  // namespace

extern "C" {

void random_ball_query_cuda_launcher(int, int, float, float, const int *,
                                     const float *, const float *, const int *,
                                     const int *, int *, float *) {
    missing_pointops_kernel("random_ball_query_cuda_launcher");
}

void farthest_point_sampling_cuda_launcher(int, int, const float *, const int *,
                                           const int *, float *, int *) {
    missing_pointops_kernel("farthest_point_sampling_cuda_launcher");
}

void grouping_forward_cuda_launcher(int, int, int, const float *, const int *, float *) {
    missing_pointops_kernel("grouping_forward_cuda_launcher");
}

void grouping_backward_cuda_launcher(int, int, int, const float *, const int *, float *) {
    missing_pointops_kernel("grouping_backward_cuda_launcher");
}

void interpolation_forward_cuda_launcher(int, int, int, const float *, const int *,
                                         const float *, float *) {
    missing_pointops_kernel("interpolation_forward_cuda_launcher");
}

void interpolation_backward_cuda_launcher(int, int, int, const float *, const int *,
                                          const float *, float *) {
    missing_pointops_kernel("interpolation_backward_cuda_launcher");
}

void subtraction_forward_cuda_launcher(int, int, int, const float *, const float *,
                                       const int *, float *) {
    missing_pointops_kernel("subtraction_forward_cuda_launcher");
}

void subtraction_backward_cuda_launcher(int, int, int, const int *, const float *,
                                        float *, float *) {
    missing_pointops_kernel("subtraction_backward_cuda_launcher");
}

void aggregation_forward_cuda_launcher(int, int, int, int, const float *, const float *,
                                       const float *, const int *, float *) {
    missing_pointops_kernel("aggregation_forward_cuda_launcher");
}

void aggregation_backward_cuda_launcher(int, int, int, int, const float *, const float *,
                                        const float *, const int *, const float *,
                                        float *, float *, float *) {
    missing_pointops_kernel("aggregation_backward_cuda_launcher");
}

void attention_relation_step_forward_cuda_launcher(int, int, int, const float *,
                                                   const float *, const float *,
                                                   const int *, const int *, float *) {
    missing_pointops_kernel("attention_relation_step_forward_cuda_launcher");
}

void attention_relation_step_backward_cuda_launcher(int, int, int, const float *, float *,
                                                    const float *, float *, const float *,
                                                    float *, const int *, const int *,
                                                    const float *) {
    missing_pointops_kernel("attention_relation_step_backward_cuda_launcher");
}

void attention_fusion_step_forward_cuda_launcher(int, int, int, const float *,
                                                 const float *, const int *,
                                                 const int *, float *) {
    missing_pointops_kernel("attention_fusion_step_forward_cuda_launcher");
}

void attention_fusion_step_backward_cuda_launcher(int, int, int, const float *, float *,
                                                  const float *, float *, const int *,
                                                  const int *, const float *) {
    missing_pointops_kernel("attention_fusion_step_backward_cuda_launcher");
}

}
