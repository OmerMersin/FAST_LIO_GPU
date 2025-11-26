#include "gpu/gpu_voxel_map.hpp"

#ifdef FASTLIO_USE_CUDA

#include "gpu/fastlio_cuda_utils.hpp"

#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cmath>
#include <vector>
#include <cstdint>

namespace fastlio
{
namespace gpu
{
namespace
{
struct PackedPoint
{
  float x;
  float y;
  float z;
  float intensity;
  float curvature;
};

__device__ inline uint64_t encode_key(int ix, int iy, int iz)
{
  const int bias = 1 << 20;
  const uint64_t mask = (1ULL << 21) - 1ULL;
  const uint64_t ux = (static_cast<uint64_t>(ix + bias) & mask);
  const uint64_t uy = (static_cast<uint64_t>(iy + bias) & mask);
  const uint64_t uz = (static_cast<uint64_t>(iz + bias) & mask);
  return (ux << 42) | (uy << 21) | uz;
}

__global__ void compute_keys_kernel(const PackedPoint *points,
                                    uint64_t *keys,
                                    int *indices,
                                    size_t total_points,
                                    float inv_leaf)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_points)
  {
    return;
  }

  const PackedPoint p = points[idx];
  const float scaled_x = p.x * inv_leaf;
  const float scaled_y = p.y * inv_leaf;
  const float scaled_z = p.z * inv_leaf;
  const int ix = static_cast<int>(floorf(scaled_x));
  const int iy = static_cast<int>(floorf(scaled_y));
  const int iz = static_cast<int>(floorf(scaled_z));
  keys[idx] = encode_key(ix, iy, iz);
  indices[idx] = static_cast<int>(idx);
}

__global__ void gather_points_kernel(const PackedPoint *input,
                                     const int *indices,
                                     PackedPoint *output,
                                     size_t count)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
  {
    return;
  }
  output[idx] = input[indices[idx]];
}
}

struct VoxelDownsampler::Impl
{
  Impl()
  {
    if (cudaStreamCreate(&stream_) != cudaSuccess)
    {
      available_ = false;
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "Failed to create CUDA stream for voxel downsampler");
    }
  }

  ~Impl()
  {
    release();
    if (stream_ != nullptr)
    {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  void set_leaf(float leaf)
  {
    if (leaf <= 0.0f)
    {
      leaf = 0.1f;
    }
    leaf_size_ = leaf;
    inv_leaf_size_ = 1.0f / leaf_size_;
  }

  bool available() const { return available_; }

  bool filter(const PointCloudXYZI &input, PointCloudXYZI &output)
  {
    if (!available_)
    {
      return false;
    }

    const size_t total_points = input.size();
    if (total_points == 0)
    {
      output.clear();
      return true;
    }

    try
    {
      ensure_capacity(total_points);
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "Voxel GPU allocation failed: %s", ex.what());
      available_ = false;
      return false;
    }

    host_points_.resize(total_points);
    for (size_t i = 0; i < total_points; ++i)
    {
      const PointType &pt = input.points[i];
      host_points_[i] = PackedPoint{pt.x, pt.y, pt.z, pt.intensity, pt.curvature};
    }

    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_points_, host_points_.data(), total_points * sizeof(PackedPoint), cudaMemcpyHostToDevice, stream_));

    const int threads = 256;
    const int blocks = static_cast<int>((total_points + threads - 1) / threads);
    compute_keys_kernel<<<blocks, threads, 0, stream_>>>(d_points_, d_keys_, d_indices_, total_points, inv_leaf_size_);
    FASTLIO_CUDA_CHECK(cudaGetLastError());

    auto exec = thrust::cuda::par.on(stream_);
    auto keys_begin = thrust::device_pointer_cast(d_keys_);
    auto idx_begin = thrust::device_pointer_cast(d_indices_);
    thrust::sort_by_key(exec, keys_begin, keys_begin + total_points, idx_begin);
    auto new_end = thrust::unique_by_key(exec, keys_begin, keys_begin + total_points, idx_begin);
    size_t unique_count = new_end.first - keys_begin;
    if (unique_count == 0)
    {
      FASTLIO_CUDA_CHECK(cudaStreamSynchronize(stream_));
      output.clear();
      return true;
    }

    const int gather_blocks = static_cast<int>((unique_count + threads - 1) / threads);
    gather_points_kernel<<<gather_blocks, threads, 0, stream_>>>(d_points_, d_indices_, d_compacted_, unique_count);
    FASTLIO_CUDA_CHECK(cudaGetLastError());

    host_compacted_.resize(unique_count);
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(host_compacted_.data(), d_compacted_, unique_count * sizeof(PackedPoint), cudaMemcpyDeviceToHost, stream_));
    FASTLIO_CUDA_CHECK(cudaStreamSynchronize(stream_));

    output.clear();
    output.reserve(unique_count);
    for (size_t i = 0; i < unique_count; ++i)
    {
      PointType pt;
      pt.x = host_compacted_[i].x;
      pt.y = host_compacted_[i].y;
      pt.z = host_compacted_[i].z;
      pt.intensity = host_compacted_[i].intensity;
      pt.curvature = host_compacted_[i].curvature;
      pt.normal_x = pt.normal_y = pt.normal_z = 0.0f;
      output.push_back(pt);
    }

    return true;
  }

private:
  void ensure_capacity(size_t desired)
  {
    if (desired <= capacity_)
    {
      return;
    }
    release();
    capacity_ = desired;
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_points_, capacity_ * sizeof(PackedPoint)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_compacted_, capacity_ * sizeof(PackedPoint)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_keys_, capacity_ * sizeof(uint64_t)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_indices_, capacity_ * sizeof(int)));
  }

  void release()
  {
    if (d_points_)
    {
      cudaFree(d_points_);
      d_points_ = nullptr;
    }
    if (d_compacted_)
    {
      cudaFree(d_compacted_);
      d_compacted_ = nullptr;
    }
    if (d_keys_)
    {
      cudaFree(d_keys_);
      d_keys_ = nullptr;
    }
    if (d_indices_)
    {
      cudaFree(d_indices_);
      d_indices_ = nullptr;
    }
    capacity_ = 0;
  }

  cudaStream_t stream_{nullptr};
  bool available_{true};
  float leaf_size_{0.1f};
  float inv_leaf_size_{10.0f};
  size_t capacity_{0};

  PackedPoint *d_points_{nullptr};
  PackedPoint *d_compacted_{nullptr};
  uint64_t *d_keys_{nullptr};
  int *d_indices_{nullptr};

  std::vector<PackedPoint> host_points_;
  std::vector<PackedPoint> host_compacted_;
};

VoxelDownsampler::VoxelDownsampler() : impl_(std::make_unique<Impl>()) {}
VoxelDownsampler::~VoxelDownsampler() = default;

void VoxelDownsampler::set_leaf_size(float leaf)
{
  if (impl_)
  {
    impl_->set_leaf(leaf);
  }
}

bool VoxelDownsampler::available() const
{
  return impl_ && impl_->available();
}

bool VoxelDownsampler::filter(const PointCloudXYZI &input, PointCloudXYZI &output)
{
  if (!impl_)
  {
    return false;
  }
  return impl_->filter(input, output);
}

}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
