#pragma once

#ifdef FASTLIO_USE_CUDA

#include <memory>
#include <vector>
#include "preprocess.h"

namespace fastlio
{
namespace gpu
{
class FeatureExtractor
{
public:
  FeatureExtractor();
  ~FeatureExtractor();

  FeatureExtractor(const FeatureExtractor &) = delete;
  FeatureExtractor &operator=(const FeatureExtractor &) = delete;

  bool available() const;

  /**
   * @brief Compute range/distance statistics for every ring using CUDA.
   *
   * @param pl_buff ring-separated points
   * @param typess per-ring features to populate; sizes are adjusted automatically
   * @param num_rings number of valid rings in pl_buff
   * @return true if work was executed on the GPU
   */
  bool compute(PointCloudXYZI (&pl_buff)[128], std::vector<orgtype> (&typess)[128], int num_rings);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gpu
}  // namespace fastlio

#endif  // FASTLIO_USE_CUDA
