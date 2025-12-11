#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "kernels.h"

namespace fs = std::filesystem;

struct Image {
    int w, h;
    std::vector<unsigned char> data;
};

Image readPPM(const std::string& file) {
    std::ifstream f(file);
    std::string magic;
    f >> magic;

    int w, h, maxv;
    f >> w >> h >> maxv;

    Image img{w, h};
    img.data.resize(w * h * 3);

    for (int i = 0; i < w * h * 3; i++)
        f >> img.data[i];

    return img;
}

void writePPM(const std::string& file, const Image& img, const unsigned char* data) {
    std::ofstream f(file);
    f << "P3\n" << img.w << " " << img.h << "\n255\n";

    for (int i = 0; i < img.w * img.h * 3; i++)
        f << (int)data[i] << " ";
}

int main(int argc, char** argv) {
    std::string kernel = "blur";
    std::string inputDir = "data/input";
    std::string outputDir = "results";
    int block = 16;

    for (int i=1; i<argc; i++) {
        if (std::string(argv[i]) == "-k") kernel = argv[++i];
        if (std::string(argv[i]) == "-i") inputDir = argv[++i];
        if (std::string(argv[i]) == "-o") outputDir = argv[++i];
        if (std::string(argv[i]) == "-b") block = atoi(argv[++i]);
    }

    fs::create_directories(outputDir);

    for (auto& p : fs::directory_iterator(inputDir)) {
        if (p.path().extension() != ".ppm") continue;

        Image img = readPPM(p.path().string());

        unsigned char *d_in, *d_out;

        cudaMalloc(&d_in, img.data.size());
        cudaMalloc(&d_out, img.data.size());

        cudaMemcpy(d_in, img.data.data(), img.data.size(), cudaMemcpyHostToDevice);

        if (kernel == "blur")
            launchBlurKernel(d_in, d_out, img.w, img.h, block);
        else
            launchSobelKernel(d_in, d_out, img.w, img.h, block);

        std::vector<unsigned char> result(img.data.size());
        cudaMemcpy(result.data(), d_out, img.data.size(), cudaMemcpyDeviceToHost);

        std::string outFile = outputDir + "/" + p.path().stem().string() + "_" + kernel + ".ppm";
        writePPM(outFile, img, result.data());

        cudaFree(d_in);
        cudaFree(d_out);

        std::cout << "Processed: " << p.path().string() << " → " << outFile << "\n";
    }

    return 0;
}
