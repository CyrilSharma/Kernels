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

// Note: input, output are all device pointers to float32 arrays
int main() {
  int N = 5000;
  vector<float> A(N), B(N + 1);
  for (int i = 0; i < N; ++i) {
    A[i] += 2 * i;
  }

  const int kernel_size = 2;
  const int stride = 1;
  const int padding = 1;
  solution2(A.data(), kernel_size, stride, padding, B.data(), A.size());
  for (auto el : B) {
    std::cout << el << " ";
  }
  std::cout << '\n';
}
