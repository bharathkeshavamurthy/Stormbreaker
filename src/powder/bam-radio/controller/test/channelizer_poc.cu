// (c) Tomohiro Arakawa (tarakawa@purdue.edu)

#include <math.h>
#include <iostream>

__device__ inline float2 ComplexScale(float2 a, float s) {
  float2 c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

__device__ inline float2 ComplexMul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

__global__ void shiftFreq(unsigned int n, float f_normalized, float2 *in,
                          float2 *out) {
  unsigned int offset = threadIdx.x;
  unsigned int stride = blockDim.x;
  for (unsigned int i = offset; i < n; i += stride) {
    float2 shift_coefficient;
    shift_coefficient.x = cospif(2.0 * f_normalized * i);
    shift_coefficient.y = sinpif(2.0 * f_normalized * i);
    out[i] = ComplexMul(in[i], shift_coefficient);
  }
}

__global__ void calcCorrelationVal(unsigned int n, unsigned int decimation,
                                   float *filter, unsigned int filt_len,
                                   float2 *in, float2 *out) {
  unsigned int in_offset = decimation * threadIdx.x;
  unsigned int in_stride = decimation * blockDim.x;
  unsigned int out_offset = threadIdx.x;
  unsigned int out_stride = blockDim.x;

  for (unsigned int i = 0; (in_offset + in_stride * i) < n; i++) {
    unsigned int inpos = in_offset + i * in_stride;
    float2 accum = make_float2(0, 0);
    for (unsigned int j = 0; j < filt_len; ++j) {
      if (inpos + j > n) {
        break;
      }
      float2 mult = ComplexScale(in[inpos + j], filter[j]);
      accum.x += mult.x;
      accum.y += mult.y;
    }
    out[out_offset + i * out_stride] = accum;
  }
}

int main(void) {
  // Number of input samples
  int N = 1 << 20;

  // Decimation factor
  int decimation = 40;

  // Allocate memory
  float2 *inData;
  cudaMallocManaged(&inData, N * sizeof(float2));
  float2 *tmpBuf;
  cudaMalloc(&tmpBuf, N * sizeof(float2));
  float2 *outData;
  cudaMallocManaged(&outData, (N / decimation + 1) * sizeof(float2));

  // Filter
  int filt_len = 200;
  float *filter;
  cudaMallocManaged(&filter, filt_len * sizeof(float));

  // Input data
  for (int i = 0; i < N; i++) {
    inData[i].x = 1.0f;
    inData[i].y = 0.0f;
  }

  // Filter
  for (int i = 0; i < filt_len; i++) {
    filter[i] = 0.1f;
  }

  // Run kernel
  shiftFreq<<<1, 256>>>(N, 0.5, inData, tmpBuf);
  cudaDeviceSynchronize();
  calcCorrelationVal<<<1, 256>>>(N, decimation, filter, filt_len, tmpBuf,
                                 outData);
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(inData);
  cudaFree(tmpBuf);
  cudaFree(outData);

  return 0;
}
