#include "../src/avg-pool-1d.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

int main() {
  // Profile with 1 million elements
  int N = 1000000;
  const int kernel_size = 3;
  const int stride = 1;
  const int padding = 0;
  const int M = ((N + 2 * padding - kernel_size) / stride) + 1;

  std::cout << "Profiling Average Pooling 1D Kernel" << std::endl;
  std::cout << "Input size: " << N << " elements" << std::endl;
  std::cout << "Output size: " << M << " elements" << std::endl;
  std::cout << "Kernel size: " << kernel_size << std::endl;
  std::cout << "Stride: " << stride << std::endl;
  std::cout << "Padding: " << padding << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // Allocate host memory
  std::vector<float> A(N);
  std::vector<float> B(M);

  // Initialize input data
  for (int i = 0; i < N; ++i) {
    A[i] = 2.0f * i;
  }

  // Warm up GPU
  std::cout << "Warming up GPU..." << std::endl;
  solve(A.data(), B.data(), N, M, kernel_size);
  cudaDeviceSynchronize();

  // Profile kernel execution
  const int num_runs = 100;
  std::vector<float> times_ms;

  std::cout << "Running kernel " << num_runs << " times for profiling..." << std::endl;

  for (int run = 0; run < num_runs; ++run) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Run the kernel
    solve(A.data(), B.data(), N, M, kernel_size);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    times_ms.push_back(elapsed_ms);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Calculate statistics
  float total_time = 0.0f;
  float min_time = times_ms[0];
  float max_time = times_ms[0];

  for (float time : times_ms) {
    total_time += time;
    min_time = std::min(min_time, time);
    max_time = std::max(max_time, time);
  }

  float avg_time = total_time / num_runs;

  // Calculate throughput
  float throughput_gbps = (N * sizeof(float) * 2) / (avg_time * 1e-3) / 1e9; // Read + Write

  // Print results
  std::cout << "\nProfiling Results:" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Number of runs: " << num_runs << std::endl;
  std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;
  std::cout << "Minimum kernel time: " << min_time << " ms" << std::endl;
  std::cout << "Maximum kernel time: " << max_time << " ms" << std::endl;
  std::cout << "Throughput: " << throughput_gbps << " GB/s" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // Verify correctness with a small test
  std::cout << "\nVerifying correctness..." << std::endl;
  std::vector<float> B_ref(M);
  reference_avg(A.data(), N, kernel_size, stride, padding, B_ref.data());
  
  if (almost_equal(B.data(), B_ref.data(), M)) {
    std::cout << "✓ Verification PASSED!" << std::endl;
  } else {
    std::cout << "✗ Verification FAILED!" << std::endl;
    return 1;
  }

  return 0;
}
