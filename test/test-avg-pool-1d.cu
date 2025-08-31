#include "../src/avg-pool-1d.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
  int N = 1000000;
  const int kernel_size = 3;
  const int stride = 1;
  const int padding = 0;
  const int M = ((N + 2 * padding - kernel_size) / stride) + 1;

  std::vector<float> A(N);
  std::vector<float> B(M);
  std::vector<float> B_ref(M);

  for (int i = 0; i < N; ++i) {
    A[i] = 2.0f * i;
  }

  // For now, we will ignore stride and padding as they complicate things.
  solve(A.data(), B.data(), N, M, kernel_size);
  reference_avg(A.data(), N, kernel_size, stride, padding, B_ref.data());

  // Print both
  // printVector(A, "Input A");
  // printVector(B, "Output B");
  // printVector(B_ref, "Reference B");

  // Verify
  if (almost_equal(B.data(), B_ref.data(), M)) {
    std::cout << "Verification PASSED!" << std::endl;
    return 0;
  } else {
    std::cerr << "Verification FAILED!" << std::endl;
    return 1;
  }
}