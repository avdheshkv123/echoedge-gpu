#include "kernels.h"
#include <cstdio>

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    __shared__ unsigned char tile[18][18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    int lx = tx + 1;
    int ly = ty + 1;

    int idx = (y * w + x) * 3;

    if (x < w && y < h) {
        tile[ly][lx] = in[idx];
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    int sum = 0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++)
            sum += tile[ly + j][lx + i];

    out[idx] = sum / 9;      // grayscale blur
    out[idx+1] = sum / 9;
    out[idx+2] = sum / 9;
}

__global__ void sobelKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int idx = (y * w + x) * 3;
    int g =
        abs(in[idx] - in[idx + 3]) +
        abs(in[idx] - in[idx + w * 3]);

    g = min(255, g);

    out[idx] = out[idx+1] = out[idx+2] = g;
}

// launcher wrappers
void launchBlurKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int block) {
    dim3 threads(block, block);
    dim3 grid((w + block - 1) / block, (h + block - 1) / block);
    blurKernel<<<grid, threads>>>(d_in, d_out, w, h);
}

void launchSobelKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int block) {
    dim3 threads(block, block);
    dim3 grid((w + block - 1) / block, (h + block - 1) / block);
    sobelKernel<<<grid, threads>>>(d_in, d_out, w, h);
}
