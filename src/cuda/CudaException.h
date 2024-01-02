#ifndef CARACAL_CUDA_CUDA_EXCEPTION_H
#define CARACAL_CUDA_CUDA_EXCEPTION_H

#include <cstddef>
#include <exception>
#include <string>

#include <cuda_runtime.h>

namespace caracal {

class CudaException : public std::exception {
public:
  CudaException(cudaError_t error, std::string &&message);
  virtual ~CudaException();

  cudaError_t GetError() const noexcept;

  virtual const char *what() const noexcept;

  static std::string BuildMessage(const char *filename,
                                  size_t line_number,
                                  const char *function_name,
                                  cudaError_t error);

private:
  cudaError_t error;
  std::string message;
};

} // namespace caracal

#define CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error)                           \
  do {                                                                         \
    if ((error) == cudaSuccess) {                                              \
      break;                                                                   \
    }                                                                          \
    throw caracal::CudaException((error),                                      \
                                 caracal::CudaException::BuildMessage(         \
                                     __FILE__, __LINE__, __func__, error));    \
  } while (0)

#endif