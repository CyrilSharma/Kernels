#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

void solution1(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  int cnt = 0;
  for (int pos = -padding; pos <= (((int)H) + padding - kernel_size);
       pos += stride) {
    float avg = 0.0;
    for (int j = 0; j < kernel_size; ++j) {
      if (pos + j >= 0 && pos + j < H)
        avg += input[pos + j];
    }
    avg /= kernel_size;
    output[cnt++] = avg;
  }
}

__global__ void solution2Kernel(const float *input, int kernel_size, int stride,
                                int padding, float *output, size_t H,
                                int H_OUT) {
  int total_threads = gridDim.x * blockDim.x;
  for (int loc = threadIdx.x; loc < H_OUT; loc += total_threads) {
    int pos = (loc * stride) - padding;
    float avg = 0.0;
    for (int j = 0; j < kernel_size; ++j) {
      if (pos + j >= 0 && pos + j < (int)H) {
        avg += input[pos + j];
      }
    }
    avg /= kernel_size;
    output[loc] = avg;
  }
}

void solution2(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  const int HSIGNED = (int)(H);
  const int HOUT = ((HSIGNED + 2 * padding - kernel_size) / stride) + 1;

  float *d_input = nullptr, *d_output = nullptr;
  cudaMalloc(&d_input, H * sizeof(float));
  cudaMalloc(&d_output, HOUT * sizeof(float));

  cudaMemcpy(d_input, input, H * sizeof(float), cudaMemcpyHostToDevice);

  // 3) Launch kernel
  solution2Kernel<<<1, 512>>>(d_input, kernel_size, stride, padding, d_output,
                              H, HOUT);

  // 4) Synchronize & check errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(output, d_output, HOUT * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

__global__ void solution3Kernel(const float *input, int kernel_size, int stride,
                                int padding, float *output, size_t H,
                                int H_OUT) {
  int total_threads = gridDim.x * blockDim.x;
  int slots_owned = (H_OUT + total_threads - 1) / total_threads;
  for (int i = 0; i < slots_owned; ++i) {
    int loc = (slots_owned * threadIdx.x) + i;
    if (loc > H_OUT)
      return;
    int pos = (loc * stride) - padding;
    float avg = 0.0;
    for (int j = 0; j < kernel_size; ++j) {
      if (pos + j >= 0 && pos + j < (int)H) {
        avg += input[pos + j];
      }
    }
    avg /= kernel_size;
    output[loc] = avg;
  }
}

void solution3(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  const int HSIGNED = (int)(H);
  const int HOUT = ((HSIGNED + 2 * padding - kernel_size) / stride) + 1;

  float *d_input = nullptr, *d_output = nullptr;
  cudaMalloc(&d_input, H * sizeof(float));
  cudaMalloc(&d_output, HOUT * sizeof(float));

  cudaMemcpy(d_input, input, H * sizeof(float), cudaMemcpyHostToDevice);

  // 3) Launch kernel
  solution3Kernel<<<1, 512>>>(d_input, kernel_size, stride, padding, d_output,
                              H, HOUT);

  // 4) Synchronize & check errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(output, d_output, HOUT * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

__global__ void solution4Kernel(const float *input, int kernel_size, int stride,
                                int padding, float *output, size_t H,
                                int H_OUT) {
  extern __shared__ float tile[];

  bool overlap = true; // (stride < kernel_size);
  int tileSize = (blockDim.x * stride) + (kernel_size - 1);
  int outPos = blockDim.x * blockIdx.x;
  int inPos = (outPos * stride) - padding;
  if (overlap) {
    for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
      int curPos = inPos + i;
      tile[i] = (curPos >= 0 && curPos < H) ? input[curPos] : 0.0f;
    }
  }
  // TODO: pack things contiguously if the stride is larger then the kernel.
  __syncthreads();

  float sum = 0.0;
  int pos = threadIdx.x * (overlap ? (stride) : (kernel_size));
  for (int j = 0; j < kernel_size; ++j) {
    if (pos + j >= 0 && pos + j < H) {
      sum += tile[pos + j];
    }
  }
  output[blockDim.x * blockIdx.x + threadIdx.x] = (sum / kernel_size);
}

void solution4(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  const int HSIGNED = (int)(H);
  const int HOUT = ((HSIGNED + 2 * padding - kernel_size) / stride) + 1;
  const int threadsPerBlock = 512;
  const int numBlocks = (HOUT + threadsPerBlock - 1) / threadsPerBlock;

  // This is great if the stride is small relative to the kernel.
  // It does a lot of extra work if the stride is large relative to the kernel.
  const int sharedBytes =
      (threadsPerBlock * stride + kernel_size - 1) * sizeof(float);

  float *d_input = nullptr, *d_output = nullptr;
  cudaMalloc(&d_input, H * sizeof(float));
  cudaMalloc(&d_output, HOUT * sizeof(float));

  cudaMemcpy(d_input, input, H * sizeof(float), cudaMemcpyHostToDevice);

  // 3) Launch kernel
  solution4Kernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(
      d_input, kernel_size, stride, padding, d_output, H, HOUT);

  // 4) Synchronize & check errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(output, d_output, HOUT * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

#define padGet(A, sz, x) ((x) < (sz)) ? (A[(x)]) : 0
__global__ void solution5Kernel(const float *input, int kernel_size, int stride,
                                int padding, float *output, size_t H,
                                int H_OUT) {
  int slots_owned = kernel_size;
  int outputPos = ((blockIdx.x * blockDim.x) + threadIdx.x) * slots_owned;
  int inputPos = (outputPos * stride) - padding;
  if (outputPos >= H_OUT)
    return;

  float sum = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    sum += padGet(input, H, (inputPos + i));
  }
  output[outputPos] = sum / kernel_size;

  int cnt = 0;
  for (int i = 1; i < slots_owned; ++i) {
    if (outputPos + i >= H_OUT)
      return;
    for (int j = 0; j < stride; ++j) {
      sum += padGet(input, H, inputPos + kernel_size + cnt);
      sum -= padGet(input, H, inputPos + cnt);
      ++cnt;
    }
    output[outputPos + i] = sum / kernel_size;
  }
}

void solution5(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  const int HSIGNED = (int)(H);
  const int HOUT = ((HSIGNED + 2 * padding - kernel_size) / stride) + 1;
  const int threadsPerBlock = 1024;
  const int outputsPerBlock = threadsPerBlock * kernel_size;
  const int numBlocks = (HOUT + outputsPerBlock - 1) / outputsPerBlock;
  float *d_input = nullptr, *d_output = nullptr;

  cudaMalloc(&d_input, H * sizeof(float));
  cudaMalloc(&d_output, HOUT * sizeof(float));
  cudaMemcpy(d_input, input, H * sizeof(float), cudaMemcpyHostToDevice);

  // 3) Launch kernel
  solution5Kernel<<<numBlocks, threadsPerBlock>>>(d_input, kernel_size, stride,
                                                  padding, d_output, H, HOUT);

  // 4) Synchronize & check errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after kernel: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(output, d_output, HOUT * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

template <typename T> void printVector(vector<T> &a) {
  for (auto el : a)
    std::cout << el << " ";
  std::cout << '\n';
}

// Note: input, output are all device pointers to float32 arrays
int main() {
  int N = 5;
  const int kernel_size = 2;
  const int stride = 2;
  const int padding = 1;
  const int M = ((N + 2 * padding - kernel_size) / stride) + 1;
  vector<float> A(N), B(M);
  for (int i = 0; i < N; ++i) {
    A[i] += 2 * i;
  }

  solution5(A.data(), kernel_size, stride, padding, B.data(), A.size());
  printVector(A);
  printVector(B);
}
