#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

// Reference verifier: compute 1D average pooling with given kernel, stride,
// padding
template <typename T>
void reference_avg(const T *A, int N, int kernel_size, int stride, int padding,
                   T *B_ref) {
  int M = ((N + 2 * padding - kernel_size) / stride) + 1;
  for (int j = 0; j < M; ++j) {
    T sum = 0;
    for (int k = 0; k < kernel_size; ++k) {
      int idx = j * stride + k - padding;
      if (idx >= 0 && idx < N) {
        sum += A[idx];
      }
    }
    B_ref[j] = sum / static_cast<T>(kernel_size);
  }
}

// Compare two float arrays with tolerance
template <typename T>
bool almost_equal(const T *a, const T *b, int len,
                  T tol = static_cast<T>(1e-3)) {
  for (int i = 0; i < len; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      std::cerr << "Mismatch at index " << i << ": expected " << b[i]
                << ", got " << a[i] << ", diff " << (b[i] - a[i]) << "\n";
      return false;
    }
  }
  return true;
}

void printVector(const std::vector<float> &v, const char *name) {
  std::cout << name << ": ";
  for (auto x : v)
    std::cout << x << ' ';
  std::cout << '\n';
}

#define padGet(A, sz, x) ((x) < (sz)) ? (A[(x)]) : 0
__global__ void kernel(const float *input, float *partialSums, float *output,
                       int I, int K, int B) {
  // We assume effectively infinite threads. Depth is the amount of work each
  // thread would have to do. Naive: NK work, depth of K Chunk array by K: N
  // work, depth of K Prefix: N work, depth of log(N)... but accuracy issues,
  // and hard to implement. Chunk array by C: N + (K / C)(N / C) work, depth of
  // max(C, K / C) Under the constraints, we always have enough threads. Can
  // just choose C = sqrt(K) Chunk array by sqrt(K): N work, depth of sqrt(K)
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int nbuckets = (I + B - 1) / B;
  if (id >= nbuckets)
    return;

  int bstart = (id * B);
  int bend = min(bstart + B - 1, I - 1);
  // printf("id := %d, %d, %d, %d, %d, %d - %d\n", id, I, B, K, nbuckets,
  // bstart, bend);

  partialSums[id] = 0;
  for (int i = bstart; i <= bend; ++i) {
    partialSums[id] += input[i];
  }

  __syncthreads();

  if (id * B >= (I - K + 1))
    return;

  // sum of input[bstart:bstart+K]
  // partialSums[i] is the sum of input[i*B:i*(B+1)]
  float sum = 0;
  int cur = bstart;
  int dest = bstart + K;
  while (cur < dest) {
    if ((dest - cur < B) || (cur % B != 0)) {
      sum += input[cur];
      // printf("1- %d, %d, %f\n", cur, dest, sum);
      cur += 1;
    } else {
      sum += partialSums[cur / B];
      // printf("2- %d, %d, %f\n", cur, dest, sum);
      cur += B;
    }
  }

  // output[i] is the sum of input[i:i+K]
  int nend = min(bend, I + K - 1);
  for (int i = bstart; i <= nend; ++i) {
    output[i] = sum / K;
    sum -= input[i];
    sum += input[i + K];
  }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the
// GPU)
void solve(const float *input, float *output, int input_size, int output_size,
           int kernel_size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  int B = min(kernel_size, 30);
  float *input_gpu;
  float *output_gpu;
  float *partialSums;
  size_t partialSize = (input_size + B - 1) / B;

  cudaMalloc((void **)&partialSums, partialSize * sizeof(float));
  cudaMalloc((void **)&input_gpu, input_size * sizeof(float));
  cudaMalloc((void **)&output_gpu, output_size * sizeof(float));
  cudaMemcpy(input_gpu, input, input_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(output_gpu, output, output_size * sizeof(float),
             cudaMemcpyHostToDevice);

  kernel<<<blocksPerGrid, threadsPerBlock>>>(input_gpu, partialSums, output_gpu,
                                             input_size, kernel_size, B);
  cudaDeviceSynchronize();

  float *h_partialSums = (float *)malloc(partialSize * sizeof(float));
  cudaMemcpy(h_partialSums, partialSums, partialSize * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(output, output_gpu, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // for (int i = 0; i < partialSize; ++i)
  //   printf("p[%d] = %f\n", i, h_partialSums[i]);
}