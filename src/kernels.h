#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

void launchBlurKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int block);
void launchSobelKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int block);

#endif
