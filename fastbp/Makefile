all: clean release

BUILD_DIR=./build
INCLUDE_DIRS=-Iextern/ -Iextern/cuCollections/include -Iextern/NVTX/c/include/
SOURCES=rasterizer.cu utils.cu
CXXFLAGS=-std=c++20 -arch=sm_86 -gencode arch=compute_86,code=sm_86 --expt-extended-lambda --expt-relaxed-constexpr --shared -Xcompiler -fPIC -o $(BUILD_DIR)/librasterizer.so

debug: clean
	mkdir $(BUILD_DIR)
	nvcc $(INCLUDE_DIRS) $(CXXFLAGS) -DDEBUG_CUDA -g -O0 $(SOURCES)

release: clean
	mkdir $(BUILD_DIR)
	nvcc $(INCLUDE_DIRS) $(CXXFLAGS) -O2 $(SOURCES)

clean:
	rm -rf $(BUILD_DIR)