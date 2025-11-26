#pragma once

#ifdef FASTLIO_USE_CUDA

#include <memory>
#include "preprocess.h"

namespace fastlio
{
namespace gpu
{
class VoxelDownsampler
{
public:
  VoxelDownsampler();
  ~VoxelDownsampler();

  VoxelDownsampler(const VoxelDownsampler &) = delete;
  VoxelDownsampler &operator=(const VoxelDownsampler &) = delete;

  void set_leaf_size(float leaf);
  bool available() const;

  /**
   * @brief Downsample a point cloud on the GPU.
   * @return true if GPU path executed, false if it fell back to CPU.
   */
  bool filter(const PointCloudXYZI &input, PointCloudXYZI &output);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
