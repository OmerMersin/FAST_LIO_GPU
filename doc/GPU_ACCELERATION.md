# GPU Acceleration Blueprint for FAST-LIO on Jetson Orin Nano

This note outlines a practical path to move FAST-LIO's main bottlenecks onto the Orin Nano's GPU while keeping ROS 2 interfaces identical. The goal is a staged migration so we can validate numerical parity after each step.

## Current implementation snapshot (Nov 2025)
- `FASTLIO_USE_CUDA=ON` now enables:
  - CUDA-accelerated range/distance precomputation for feature extraction (Ouster path).
  - A GPU voxel downsampler that replaces the surface `VoxelGrid` filter when available.
- Both kernels allocate once and fall back to the CPU path automatically if CUDA resources are missing.
- Build flag requirements:
  ```bash
  export CUDACXX=/usr/local/cuda/bin/nvcc   # ensure nvcc is discoverable
  colcon build --packages-select fast_lio --cmake-args -DFASTLIO_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  ```
- If the toolkit is absent, CMake will exit early with `No CMAKE_CUDA_COMPILER could be found`; either install JetPack/CUDA or disable the option.

### How to verify the GPU path is active
1. **ROS 2 logs (fastest sanity check):**
  - When CUDA feature extraction kicks in you will see `CUDA feature extractor active for Ouster scans.`
  - When the voxel downsampler switches to the GPU you will see `CUDA voxel downsampler active during filtering.`
  - If either component falls back, a warning is emitted explaining why.
  - Launch FAST-LIO as usual and watch the terminal output (or pipe it through `grep CUDA`) to confirm which path is active.
2. **tegrastats (Jetson real-time monitor):**
  ```bash
  sudo tegrastats --interval 1000
  ```
  - Look for `GR3D_FREQ` rising above 0% and `VDD_CPU_GPU_CV` power bumps synchronized with FAST-LIO callbacks. The sample trace in the issue description shows GR3D hitting 90%+, confirming GPU work.
3. **Nsight Systems / nvprof (deep dive):**
  ```bash
  nsys profile --trace=cuda,osrt -o fastlio_gpu.trace ros2 launch fast_lio mapping.launch.py ...
  ```
  - GPU kernels appear as `fastlio::gpu::FeatureExtractor::compute` and `fastlio::gpu::VoxelDownsampler::filter` so you can correlate runtime with ROS callbacks.
4. **Topic-level latency:**
  - Add `--ros-args -p debug_timers:=true` (future work) or collect callback durations with `ros2 trace` to compare CPU vs GPU builds.

## 1. Profile the Baseline
- Use `nvprof` / `nsys profile` with `--trace=cuda,osrt` while running your typical bag.
- Log per-callback timing via `tracetools` (`ros2 trace -s -k callback_start`).
- Expect the heaviest CPU kernels to be:
  1. **Feature extraction / preprocessing** (hundreds of thousands of points per scan).
  2. **ikd-Tree updates** (nearest-neighbor queries dominate)
  3. **Iterated EKF linear algebra** (dense 27×27 blocks, SVDs, etc.).

## 2. Port Feature Extraction to CUDA
- Replace the scalar loops in `Preprocess::give_feature` with warp-level primitives:
  - Store per-ring points in shared memory tiles, run curvature/dist computations via cooperative groups.
  - Use thrust or custom kernels for filtering + blind-zone masking.
- Output remains a `pcl::PointCloud<PointType>`; conversions still happen on CPU to limit ROS surface changes.
- For faster development, start from [Cupoch](https://github.com/stevenlovegrove/cupoch) or `pcl::gpu` modules rather than writing kernels from scratch.

## 3. GPU-friendly KD-tree / Voxel Map
- Replace ikd-Tree with a voxel-hash map in device memory (e.g., Cupoch's `VoxelGrid` or Voxgraph-style hashed block storage).
- Implement parallel nearest-neighbor search using either:
  - NVIDIA's `cuSpatial` kNN primitives, or
  - A custom LBVH structure (see `cupoch/geometry/geometry_kdtree.cuh`).
- Keep a thin CPU stub that mirrors the GPU map metadata so existing EKF code can request correspondences without awareness of CUDA specifics.

## 4. EKF and Linear Algebra
- The predict/update steps operate on small dense matrices. Use cuBLAS or Eigen Tensor's CUDA backend:
  - Batch all measurement Jacobians for a scan and call `cublasDgemmBatched`.
  - For SVD or LDLT, rely on cuSOLVER's `gesvdjBatched` to keep numerical parity.
- Maintain fallbacks to the CPU path via an `#ifdef FASTLIO_USE_CUDA` block so Jetson Nano can run either mode at runtime.

## 5. ROS 2 Integration Details
- Build system: add an option in `CMakeLists.txt`
  ```cmake
  option(FASTLIO_USE_CUDA "Enable CUDA kernels" OFF)
  if(FASTLIO_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DFASTLIO_USE_CUDA)
    # add cuda sources via cuda_add_library or modern CMake's add_library + LANGUAGE CUDA
  endif()
  ```
- Executor: switch `fastlio_mapping` to `rclcpp::executors::MultiThreadedExecutor` (already available) so CPU callback threads stay responsive while CUDA kernels run asynchronously.
- Memory: Preallocate device buffers (point clouds, Jacobians, voxel blocks) at node startup to avoid runtime `cudaMalloc` on the Nano.

## 6. Validation Plan
1. **Unit tests**: For each CUDA kernel, add gtest comparing GPU output to CPU baseline on recorded scan snippets.
2. **Replay bags**: run `ros2 bag play` at 1×, collect `/Odometry` and compare to CPU run via `evo_traj` / `rpe` metrics.
3. **Live smoke test**: enable GPU path with `FASTLIO_USE_CUDA=ON` only after the above passes.

## 7. Performance Expectations on Orin Nano
- Feature extraction: 3–4× speedup (point-wise parallelism).
- Map correspondence: 5×+ if ikd-Tree is fully replaced by GPU voxel hash.
- EKF: smaller gains (1.5–2×) because matrices are tiny; benefit mainly from freeing CPU.
- Thermal headroom: keep kernels <70% SM occupancy to avoid throttling; consider `nvpmodel -m 2` + `jetson_clocks` when testing.

## 8. Next Steps
1. Land the CMake/CUDA scaffolding with empty kernels (compiles on all platforms).
2. Port preprocessing kernels and guard with `FASTLIO_USE_CUDA`.
3. Replace ikd-Tree with GPU hash map, keeping a CPU compatibility mode for regression testing.
4. Move EKF math to cuBLAS/cuSOLVER and benchmark convergence.
5. Document deployment steps (Dockerfile + `nvidia-container-runtime`) for the Jetson targets.

This staged approach keeps the project shippable after every milestone while giving a clear path to a fully GPU-enabled FAST-LIO variant.
