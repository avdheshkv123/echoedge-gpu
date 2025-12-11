EchoEdgeGPU – High-Performance Image Processing with CUDA
🚀 Overview

EchoEdgeGPU is a GPU-accelerated image-processing pipeline built using CUDA C++, implementing high-performance Blur and Sobel Edge Detection kernels.
The project demonstrates how GPU parallelism drastically accelerates large-scale image operations that are computationally expensive on CPUs.

This project is designed as a capstone submission for the CUDA at Scale specialization, showcasing skills in:

## CUDA kernel design

- Shared memory optimization

- Parallel image processing

- Batch GPU workflows

- Makefile-based build systems

- Reproducible pipelines

📁 Project Structure
EchoEdgeGPU/
├── README.md
├── Makefile
├── run.sh
├── run_example.sh
├── data/
│   └── input/              # sample PPM images
├── results/                # output generated
└── src/
    ├── main.cu             # main pipeline (IO + kernel launcher)
    ├── kernels.cu          # CUDA kernels (blur + sobel)
    └── kernels.h           # kernel headers

## 🧠 Features
✅ CUDA-Accelerated Blur Filter
-Uses shared memory for high performance
-3×3 convolution
-Grayscale smoothing output

✅ CUDA Sobel Edge Detection
-Computes gradients
-Highlights sharp transitions
-Parallelized across all pixels

✅ Batch Processing
-Process an entire folder of input images at once.

✅ No Dependencies
-No OpenCV
-No external libs
-Uses lightweight PPM loader
-Easily runnable in any CUDA environment

