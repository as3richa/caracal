#include "CudaException.h"

#include <cassert>
#include <cstddef>
#include <exception>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

namespace caracal {
CudaException::CudaException(cudaError_t error, std::string &&message)
    : error(error), message(move(message)) {
  assert(error != cudaSuccess);
}

CudaException::~CudaException() {}

cudaError_t CudaException::GetError() const noexcept { return error; }

const char *CudaException::what() const noexcept { return message.c_str(); }

std::string CudaException::BuildMessage(const char *filename,
                                        size_t line_number,
                                        const char *function_name,
                                        cudaError_t error) {
  std::stringstream stream;
  stream << "CUDA error at " << filename << ':' << line_number << " ("
         << function_name << "): " << cudaGetErrorName(error) << ": "
         << cudaGetErrorString(error);
  return stream.str();
}
} // namespace caracal