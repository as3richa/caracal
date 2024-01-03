CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -pedantic -O3 -flto -DNDEBUG

CUDA_CXX := nvcc
CUDA_CXXFLAGS := -dlto -DNDEBUG
CUDA_LFLAGS := -lcublas

BUILD_DIR := build

SRC_DIR := src
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_DIR := $(BUILD_DIR)/obj
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

CUDA_SRC_DIR := $(SRC_DIR)/cuda
CUDA_SRC_FILES := $(wildcard $(CUDA_SRC_DIR)/*.cu $(CUDA_SRC_DIR)/*.cpp)
CUDA_OBJ_DIR := $(BUILD_DIR)/obj/cuda
CUDA_OBJ_FILES := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRC_FILES))

CUDA_TEST_SRC_DIR := $(CUDA_SRC_DIR)/test
CUDA_TEST_SRC_FILES := $(wildcard $(CUDA_TEST_SRC_DIR)/*.cpp)
CUDA_TEST_OBJ_DIR := $(BUILD_DIR)/obj/cuda/test
CUDA_TEST_OBJ_FILES := $(patsubst $(CUDA_TEST_SRC_DIR)/%.cpp,$(CUDA_TEST_OBJ_DIR)/%.o,$(CUDA_TEST_SRC_FILES))
CUDA_TEST_BIN_DIR := $(BUILD_DIR)/test/cuda
CUDA_TEST_BIN_FILES := $(patsubst $(CUDA_TEST_SRC_DIR)/%.cpp,$(CUDA_TEST_BIN_DIR)/%,$(CUDA_TEST_SRC_FILES))
CUDA_TEST_BIN_TARGETS := $(patsubst $(CUDA_TEST_SRC_DIR)/%.cpp,cuda-test/%,$(CUDA_TEST_SRC_FILES))

TARGET_LIB := $(BUILD_DIR)/libcaracal.a

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJ_FILES) $(CUDA_OBJ_FILES)
	ar cr $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@

$(CUDA_TEST_OBJ_DIR)/%.o: $(CUDA_TEST_SRC_DIR)/%.cpp | $(CUDA_TEST_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_TEST_BIN_DIR)/%: $(TARGET_LIB) $(CUDA_TEST_OBJ_DIR)/%.o | $(CUDA_TEST_BIN_DIR)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(CUDA_LFLAGS) $^ -o $@

cuda-test: $(CUDA_TEST_OBJ_FILES) $(CUDA_TEST_BIN_FILES) $(CUDA_TEST_BIN_TARGETS)

cuda-test/%: $(CUDA_TEST_BIN_DIR)/%
	$<

$(BUILD_DIR) $(OBJ_DIR) $(CUDA_OBJ_DIR) $(CUDA_TEST_OBJ_DIR) $(CUDA_TEST_BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

format:
	clang-format -i $(shell find . -name '*.cpp' -or -name '*.cu' -or -name '*.h' )

.PHONY: all clean format cuda-test
