#include <torch/extension.h>


at::Tensor qpa_cuda_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& cam_mask,
    const at::Tensor& inv_cnt,
    const int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qpa_cuda_forward",
          &qpa_cuda_forward);
}
