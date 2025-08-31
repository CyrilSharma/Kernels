#include <cstdio>
#include <cuda_runtime.h>

// This is an excellent algorithm for the wrong problem.
// I solved the sliding window problem instead of the convolution problem...
__global__ void convolution_1d_kernel(const float *input, float *partialSums,
                                      float *output, int I, int K, int B) {
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
  for (int i = bstart; i <= bend; ++i)
    partialSums[id] += input[i];
  __syncthreads();

  if (id * B >= (I - K + 1))
    return;

  // sum of input[bstart:bstart+K]
  // partialSums[i] is the sum of input[i*B:i*(B+1)]
  int sum = 0;
  int cur = bstart;
  int dest = bstart + K;
  while (cur < dest) {
    if ((dest - cur < B) || (cur % B != 0)) {
      sum += input[cur];
      cur += 1;
    } else {
      sum += partialSums[cur / B];
      cur += B;
    }
  }

  // output[i] is the sum of input[i:i+K]
  int nend = min(bend, I + K - 1);
  for (int i = bstart; i <= nend; ++i) {
    output[i] = sum;
    sum -= input[i];
    sum += input[i + K - 1];
  }
  // hello!
}

// input, kernel, output are device pointers (i.e. pointers to memory on the
// GPU)
void solve(const float *input, float *output, int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int threadsPerBlock = 256;
  int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  int B = min(kernel_size, 30);
  float *partialSums;
  size_t partialSize = (input_size + B - 1) / B;
  cudaMalloc((void **)&partialSums, partialSize * sizeof(float));

  convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input, partialSums, output, input_size, kernel_size, B);
  cudaDeviceSynchronize();

  float *h_partialSums = (float *)malloc(partialSize * sizeof(float));
  cudaMemcpy(h_partialSums, partialSums, partialSize * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < partialSize; ++i)
    printf("p[%d] = %f\n", i, h_partialSums[i]);
}