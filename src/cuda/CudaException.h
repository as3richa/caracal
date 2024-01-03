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

#define CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR(error)                      \
  CARACAL_CUDA_EXCEPTION_THROW_ON_ERROR(cudaGetLastError())

#define CARACAL_CUDA_EXCEPTION_THROW_ON_ERROR(error)                           \
  do {                                                                         \
    cudaError_t caracal_cuda_exception_throw_on_error_error = (error);         \
    if (caracal_cuda_exception_throw_on_error_error == cudaSuccess) {          \
      break;                                                                   \
    }                                                                          \
    throw caracal::CudaException(                                              \
        caracal_cuda_exception_throw_on_error_error,                           \
        caracal::CudaException::BuildMessage(                                  \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__,                                                          \
            caracal_cuda_exception_throw_on_error_error));                     \
  } while (0)

#endif