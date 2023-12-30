CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic

CUDA_CXX := nvcc
CUDA_LDFLAGS := -lcudart -lcublas -lcurand

BUILD_DIR := build

SRC_DIR := src
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_DIR := $(BUILD_DIR)/obj
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

CUDA_SRC_DIR := $(SRC_DIR)/cuda
CUDA_SRC_FILES := $(wildcard $(CUDA_SRC_DIR)/*.cu)
CUDA_OBJ_DIR := $(BUILD_DIR)/obj/cuda
CUDA_OBJ_FILES := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRC_FILES))

TARGET_LIB := $(BUILD_DIR)/libcaracal.a

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJ_FILES) $(CUDA_OBJ_FILES)
	$(CUDA_CXX) $(CUDA_LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR)
	$(CUDA_CXX) -c $< -o $@

$(BUILD_DIR) $(OBJ_DIR) $(CUDA_OBJ_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

format:
	clang-format -i src/*.{cpp,h} src/cuda/*.{cu,h}

.PHONY: all clean format
