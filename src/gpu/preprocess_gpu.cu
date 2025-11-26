#include "gpu/preprocess_gpu.hpp"

#ifdef FASTLIO_USE_CUDA

#include "gpu/fastlio_cuda_utils.hpp"

#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <cmath>

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
  float time;
};

__global__ void compute_range_dista_kernel(const PackedPoint *points,
                                           const int *next_index,
                                           double *ranges,
                                           double *distances,
                                           size_t total_points)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_points)
  {
    return;
  }

  const PackedPoint p = points[idx];
  const double range = sqrt(static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y + static_cast<double>(p.z) * p.z);
  ranges[idx] = range;

  const int next_idx = next_index[idx];
  if (next_idx >= 0)
  {
    const PackedPoint pn = points[next_idx];
    const double dx = static_cast<double>(p.x) - static_cast<double>(pn.x);
    const double dy = static_cast<double>(p.y) - static_cast<double>(pn.y);
    const double dz = static_cast<double>(p.z) - static_cast<double>(pn.z);
    distances[idx] = dx * dx + dy * dy + dz * dz;
  }
  else
  {
    distances[idx] = 0.0;
  }
}
}  // namespace

struct FeatureExtractor::Impl
{
  Impl()
  {
    auto err = cudaStreamCreate(&stream_);
    available_ = (err == cudaSuccess);
    if (!available_)
    {
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "Failed to create CUDA stream: %s", cudaGetErrorString(err));
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

  bool available() const { return available_; }

  bool compute(PointCloudXYZI (&pl_buff)[128], std::vector<orgtype> (&typess)[128], int num_rings)
  {
    if (!available_)
    {
      return false;
    }

    size_t total_points = 0;
    for (int ring = 0; ring < num_rings; ++ring)
    {
      total_points += static_cast<size_t>(pl_buff[ring].size());
    }

    if (total_points == 0)
    {
      return false;
    }

    try
    {
      ensure_capacity(total_points);
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR(rclcpp::get_logger("fast_lio.gpu"), "CUDA allocation failed: %s", ex.what());
      available_ = false;
      return false;
    }

    host_points_.resize(total_points);
    host_next_index_.resize(total_points);

    size_t cursor = 0;
    for (int ring = 0; ring < num_rings; ++ring)
    {
      PointCloudXYZI &pl = pl_buff[ring];
      const size_t ring_size = pl.size();
      std::vector<orgtype> &types = typess[ring];
      types.clear();
      types.resize(ring_size);
      if (ring_size == 0)
      {
        continue;
      }

      for (size_t i = 0; i < ring_size; ++i)
      {
        const PointType &pt = pl.points[i];
        host_points_[cursor + i] = PackedPoint{pt.x, pt.y, pt.z, pt.curvature};
        host_next_index_[cursor + i] = (i + 1 < ring_size) ? static_cast<int>(cursor + i + 1) : -1;
      }
      cursor += ring_size;
    }

    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_points_, host_points_.data(), total_points * sizeof(PackedPoint), cudaMemcpyHostToDevice, stream_));
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(d_next_index_, host_next_index_.data(), total_points * sizeof(int), cudaMemcpyHostToDevice, stream_));

    const int threads = 256;
    const int blocks = static_cast<int>((total_points + threads - 1) / threads);
    compute_range_dista_kernel<<<blocks, threads, 0, stream_>>>(d_points_, d_next_index_, d_ranges_, d_distances_, total_points);
    FASTLIO_CUDA_CHECK(cudaGetLastError());

    host_ranges_.resize(total_points);
    host_distances_.resize(total_points);
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(host_ranges_.data(), d_ranges_, total_points * sizeof(double), cudaMemcpyDeviceToHost, stream_));
    FASTLIO_CUDA_CHECK(cudaMemcpyAsync(host_distances_.data(), d_distances_, total_points * sizeof(double), cudaMemcpyDeviceToHost, stream_));
    FASTLIO_CUDA_CHECK(cudaStreamSynchronize(stream_));

    cursor = 0;
    for (int ring = 0; ring < num_rings; ++ring)
    {
      PointCloudXYZI &pl = pl_buff[ring];
      const size_t ring_size = pl.size();
      std::vector<orgtype> &types = typess[ring];
      for (size_t i = 0; i < ring_size; ++i)
      {
        orgtype &target = types[i];
        target.range = host_ranges_[cursor + i];
        target.dista = host_distances_[cursor + i];
        target.edj[Prev] = Nr_nor;
        target.edj[Next] = Nr_nor;
        target.ftype = Nor;
        target.intersect = 2.0;
      }
      cursor += ring_size;
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
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_next_index_, capacity_ * sizeof(int)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_ranges_, capacity_ * sizeof(double)));
    FASTLIO_CUDA_CHECK(cudaMalloc(&d_distances_, capacity_ * sizeof(double)));
  }

  void release()
  {
    if (d_points_)
    {
      cudaFree(d_points_);
      d_points_ = nullptr;
    }
    if (d_next_index_)
    {
      cudaFree(d_next_index_);
      d_next_index_ = nullptr;
    }
    if (d_ranges_)
    {
      cudaFree(d_ranges_);
      d_ranges_ = nullptr;
    }
    if (d_distances_)
    {
      cudaFree(d_distances_);
      d_distances_ = nullptr;
    }
    capacity_ = 0;
  }

  cudaStream_t stream_{nullptr};
  bool available_{true};
  size_t capacity_{0};
  PackedPoint *d_points_{nullptr};
  int *d_next_index_{nullptr};
  double *d_ranges_{nullptr};
  double *d_distances_{nullptr};

  std::vector<PackedPoint> host_points_;
  std::vector<int> host_next_index_;
  std::vector<double> host_ranges_;
  std::vector<double> host_distances_;
};

FeatureExtractor::FeatureExtractor() : impl_(std::make_unique<Impl>()) {}
FeatureExtractor::~FeatureExtractor() = default;

bool FeatureExtractor::available() const
{
  return impl_ && impl_->available();
}

bool FeatureExtractor::compute(PointCloudXYZI (&pl_buff)[128], std::vector<orgtype> (&typess)[128], int num_rings)
{
  if (!impl_)
  {
    return false;
  }
  return impl_->compute(pl_buff, typess, num_rings);
}

}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
