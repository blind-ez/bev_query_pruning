#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <math.h>

#define CUDA_1D_KERNEL_LOOP(i, n)   for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < (n); i += (int)(blockDim.x * gridDim.x))

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {return (N + num_threads - 1) / num_threads;}


template <typename scalar_t>
__device__ __forceinline__ scalar_t qpa_bilinear(
  const scalar_t* __restrict__ bottom_data,
  const int height,
  const int width,
  const int nheads,
  const int channels,
  const scalar_t h,
  const scalar_t w,
  const int m,
  const int c
) {
  const int h_low = (int)floorf((float)h);
  const int w_low = (int)floorf((float)w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - (scalar_t)h_low;
  const scalar_t lw = w - (scalar_t)w_low;
  const scalar_t hh = (scalar_t)1 - lh;
  const scalar_t hw = (scalar_t)1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;

  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) {
    v1 = bottom_data[h_low * h_stride + w_low * w_stride + base_ptr];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high >= 0 && h_low < height && w_high < width) {
    v2 = bottom_data[h_low * h_stride + w_high * w_stride + base_ptr];
  }
  scalar_t v3 = 0;
  if (h_high >= 0 && w_low >= 0 && h_high < height && w_low < width) {
    v3 = bottom_data[h_high * h_stride + w_low * w_stride + base_ptr];
  }
  scalar_t v4 = 0;
  if (h_high >= 0 && w_high >= 0 && h_high < height && w_high < width) {
    v4 = bottom_data[h_high * h_stride + w_high * w_stride + base_ptr];
  }

  const scalar_t w1 = hh * hw;
  const scalar_t w2 = hh * lw;
  const scalar_t w3 = lh * hw;
  const scalar_t w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename scalar_t>
__global__ void qpa_gpu_kernel(
  const int n,
  const scalar_t* __restrict__ data_value,
  const int64_t* __restrict__ data_spatial_shapes,
  const int64_t* __restrict__ data_level_start_index,
  const scalar_t* __restrict__ data_sampling_loc,
  const scalar_t* __restrict__ data_attn_weight,
  const uint8_t* __restrict__ data_cam_mask,
  const float* __restrict__ data_inv_cnt,
  const int batch_size,
  const int num_cams,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  scalar_t* __restrict__ data_col
) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int tmp = index;
    const int c_col = tmp % channels; tmp /= channels;
    const int m_col = tmp % num_heads; tmp /= num_heads;
    const int q_col = tmp % num_query; tmp /= num_query;
    const int b_col = tmp;

    const uint8_t mask = data_cam_mask[b_col * num_query + q_col];
    const float inv = data_inv_cnt[b_col * num_query + q_col];
    if (mask == 0 || inv == 0.0f) {
      data_col[index] = (scalar_t)0;
      continue;
    }

    const int qid_stride = num_heads * channels;

    const int sampling_index = (b_col * num_query + q_col) * num_heads + m_col;
    int data_weight_ptr0 = sampling_index * num_levels * num_point;

    scalar_t acc = (scalar_t)0;

    #pragma unroll
    for (int cam = 0; cam < num_cams; ++cam) {
      if (!(mask & (1u << cam))) continue;

      const int data_value_ptr_init_offset = (b_col * num_cams + cam) * spatial_size * qid_stride;

      scalar_t col_cam = (scalar_t)0;
      int data_weight_ptr = data_weight_ptr0;

      for (int l = 0; l < num_levels; ++l) {
        const int64_t level_start = data_level_start_index[l];
        const int spatial_h = (int)data_spatial_shapes[l * 2 + 0];
        const int spatial_w = (int)data_spatial_shapes[l * 2 + 1];

        const scalar_t* __restrict__ data_value_ptr = data_value + (data_value_ptr_init_offset + (int)level_start * qid_stride);

        for (int p = 0; p < num_point; ++p) {
          const int loc_base = (((((b_col * num_cams + cam) * num_query + q_col) * num_heads + m_col) * num_levels + l) * num_point + p) * 2;

          const scalar_t loc_w = data_sampling_loc[loc_base + 0];
          const scalar_t loc_h = data_sampling_loc[loc_base + 1];
          const scalar_t weight = data_attn_weight[data_weight_ptr];

          const scalar_t h_im = loc_h * (scalar_t)spatial_h - (scalar_t)0.5;
          const scalar_t w_im = loc_w * (scalar_t)spatial_w - (scalar_t)0.5;

          if (h_im > (scalar_t)-1 && w_im > (scalar_t)-1 && h_im < (scalar_t)spatial_h && w_im < (scalar_t)spatial_w) {
            col_cam += qpa_bilinear(
              data_value_ptr,
              spatial_h,
              spatial_w,
              num_heads,
              channels,
              h_im,
              w_im,
              m_col,
              c_col
            ) * weight;
          }

          data_weight_ptr += 1;
        }
      }

      acc += col_cam;
    }

    data_col[index] = acc * (scalar_t)inv;
  }
}

template <typename scalar_t>
inline void qpa_cuda(
  cudaStream_t stream,
  const scalar_t* data_value,
  const int64_t* data_spatial_shapes,
  const int64_t* data_level_start_index,
  const scalar_t* data_sampling_loc,
  const scalar_t* data_attn_weight,
  const uint8_t* data_cam_mask,
  const float* data_inv_cnt,
  const int batch_size,
  const int num_cams,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  scalar_t* data_col
) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  qpa_gpu_kernel<scalar_t> <<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
    num_kernels,
    data_value,
    data_spatial_shapes,
    data_level_start_index,
    data_sampling_loc,
    data_attn_weight,
    data_cam_mask,
    data_inv_cnt,
    batch_size,
    num_cams,
    spatial_size,
    num_heads,
    channels,
    num_levels,
    num_query,
    num_point,
    data_col
  );
}

at::Tensor qpa_cuda_forward(
  const at::Tensor& value,
  const at::Tensor& spatial_shapes,
  const at::Tensor& level_start_index,
  const at::Tensor& sampling_loc,
  const at::Tensor& attn_weight,
  const at::Tensor& cam_mask,
  const at::Tensor& inv_cnt,
  const int im2col_step
) {
  AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
  AT_ASSERTM(cam_mask.is_cuda(), "cam_mask must be a CUDA tensor");
  AT_ASSERTM(inv_cnt.is_cuda(), "inv_cnt must be a CUDA tensor");

  AT_ASSERTM(value.is_contiguous(), "value must be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index must be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight must be contiguous");
  AT_ASSERTM(cam_mask.is_contiguous(), "cam_mask must be contiguous");
  AT_ASSERTM(inv_cnt.is_contiguous(), "inv_cnt must be contiguous");

  AT_ASSERTM(spatial_shapes.scalar_type() == at::kLong, "spatial_shapes must be int64");
  AT_ASSERTM(level_start_index.scalar_type() == at::kLong, "level_start_index must be int64");
  AT_ASSERTM(cam_mask.scalar_type() == at::kByte, "cam_mask must be uint8 (Byte)");
  AT_ASSERTM(inv_cnt.scalar_type() == at::kFloat, "inv_cnt must be float32");

  AT_ASSERTM(value.dim() == 5, "value must be 5D [B, 6, S, H, C]");
  AT_ASSERTM(sampling_loc.dim() == 7, "sampling_loc must be 7D [B, 6, Q, H, L, P, 2]");
  AT_ASSERTM(attn_weight.dim() == 5, "attn_weight must be 5D [B, Q, H, L, P]");
  AT_ASSERTM(cam_mask.dim() == 2, "cam_mask must be 2D [B, Q]");
  AT_ASSERTM(inv_cnt.dim() == 2, "inv_cnt must be 2D [B, Q]");

  const int B = (int)value.size(0);
  const int V = (int)value.size(1);
  const int S = (int)value.size(2);
  const int H = (int)value.size(3);
  const int C = (int)value.size(4);

  const int L = (int)spatial_shapes.size(0);
  const int Q = (int)sampling_loc.size(2);
  const int P = (int)sampling_loc.size(5);

  AT_ASSERTM(V == 6, "V must be 6 (expected)");

  AT_ASSERTM((int)sampling_loc.size(0) == B, "sampling_loc batch mismatch");
  AT_ASSERTM((int)sampling_loc.size(1) == V, "sampling_loc cam mismatch");
  AT_ASSERTM((int)sampling_loc.size(3) == H, "sampling_loc num_heads mismatch");
  AT_ASSERTM((int)sampling_loc.size(4) == L, "sampling_loc num_levels mismatch");
  AT_ASSERTM((int)sampling_loc.size(6) == 2, "sampling_loc last dim must be 2");

  AT_ASSERTM((int)attn_weight.size(0) == B, "attn_weight batch mismatch");
  AT_ASSERTM((int)attn_weight.size(1) == Q, "attn_weight query mismatch");
  AT_ASSERTM((int)attn_weight.size(2) == H, "attn_weight num_heads mismatch");
  AT_ASSERTM((int)attn_weight.size(3) == L, "attn_weight num_levels mismatch");
  AT_ASSERTM((int)attn_weight.size(4) == P, "attn_weight P mismatch");

  AT_ASSERTM((int)cam_mask.size(0) == B && (int)cam_mask.size(1) == Q, "cam_mask shape must be [B, Q]");
  AT_ASSERTM((int)inv_cnt.size(0) == B && (int)inv_cnt.size(1) == Q, "inv_cnt shape must be [B, Q]");

  const int im2col_step_ = std::min(B, im2col_step);
  AT_ASSERTM(B % im2col_step_ == 0, "batch must be divisible by im2col_step_");

  auto out = at::zeros({B, Q, H, C}, value.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int qid_stride = H * C;

  const int per_value_size = V * S * qid_stride;
  const int per_sample_loc_size = V * Q * H * L * P * 2;
  const int per_attn_weight_size = Q * H * L * P;
  const int per_mask_size = Q;
  const int per_inv_size = Q;

  const int Bn = im2col_step_;

  auto out_n = out.view({B / im2col_step_, Bn, Q, H, C});

  for (int n = 0; n < B / im2col_step_; ++n) {
    auto cols = out_n.select(0, n);
    const int b_global_offset = n * Bn;

    AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(),
      "qpa_cuda_forward",
      [&] {
        qpa_cuda<scalar_t>(
          stream,
          value.data_ptr<scalar_t>() + b_global_offset * per_value_size,
          spatial_shapes.data_ptr<int64_t>(),
          level_start_index.data_ptr<int64_t>(),
          sampling_loc.data_ptr<scalar_t>() + b_global_offset * per_sample_loc_size,
          attn_weight.data_ptr<scalar_t>() + b_global_offset * per_attn_weight_size,
          cam_mask.data_ptr<uint8_t>() + b_global_offset * per_mask_size,
          inv_cnt.data_ptr<float>() + b_global_offset * per_inv_size,
          Bn,
          V,
          S,
          H,
          C,
          L,
          Q,
          P,
          cols.data_ptr<scalar_t>()
        );
      }
    );
  }

  return out.view({B, Q, H * C});
}
