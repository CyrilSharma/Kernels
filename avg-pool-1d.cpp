// #include <bits/stdc++.h>
#include <array>
#include <iostream>
#include <print>
using namespace std;

void solution1(const float *input, int kernel_size, int stride, int padding,
               float *output, size_t H) {
  int cnt = 0;
  println("HELLO?");
  for (int pos = -padding; pos <= (((int)H) + padding - kernel_size);
       pos += stride) {
    float avg = 0.0;
    for (int j = 0; j < kernel_size; ++j) {
      if (pos + j >= 0 && pos + j < H)
        avg += input[pos + j];
    }
    avg /= kernel_size;
    std::cout << "cnt: " << cnt << ", " << "avg: " << avg;
    output[cnt++] = avg;
  }
}

int main() {
  array<float, 5> A = {2, 4, 6, 8, 10};
  const int kernel_size = 2;
  const int stride = 1;
  const int padding = 1;
  array<float, 5> B = {0, 0, 0, 0, 0};
  solution1(A.data(), kernel_size, stride, padding, B.data(), A.size());
  println("{}", B);
}
