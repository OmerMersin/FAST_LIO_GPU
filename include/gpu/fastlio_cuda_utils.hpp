#pragma once

#ifdef FASTLIO_USE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace fastlio
{
namespace gpu
{
namespace detail
{
inline void CheckCuda(cudaError_t err, const char *expr, const char *file, int line)
{
  if (err == cudaSuccess)
  {
    return;
  }
  std::ostringstream oss;
  oss << "CUDA call '" << expr << "' failed at " << file << ":" << line << " with error '"
      << cudaGetErrorString(err) << "'";
  throw std::runtime_error(oss.str());
}
}  // namespace detail
}  // namespace gpu
}  // namespace fastlio

#define FASTLIO_CUDA_CHECK(expr) ::fastlio::gpu::detail::CheckCuda((expr), #expr, __FILE__, __LINE__)

#endif  // FASTLIO_USE_CUDA
