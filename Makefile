NVCC = nvcc
CFLAGS = -O3 -std=c++17

SRC = src/main.cu src/kernels.cu
OUT = echoedge_gpu.exe

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(OUT) results/*.ppm
