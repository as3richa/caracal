#ifndef CARACAL_CUDA_DEVICE_POINTER_H_
#define CARACAL_CUDA_DEVICE_POINTER_H_

#include <cstddef>

#include <cuda_runtime.h>

#include "CudaException.h"

namespace caracal {

template <typename T> class DevicePointer {
private:
  T *ptr;

public:
  // FIXME: nope
  DevicePointer() : ptr(nullptr) {}
  DevicePointer(T *ptr) : ptr(ptr) {}

  DevicePointer(DevicePointer<T> &&other) : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  ~DevicePointer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  DevicePointer(const DevicePointer<T> &other) = delete;
  DevicePointer<T> operator=(const DevicePointer<T> &other) = delete;
  DevicePointer<T> operator=(DevicePointer<T> &&other) = delete;

  static DevicePointer<T> Malloc(size_t count) {
    T *ptr;

    const cudaError_t error = cudaMalloc(&ptr, count * sizeof(T));
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);

    return DevicePointer(ptr);
  }

  static DevicePointer<T> Memcpy(const T *source, size_t count) {
    DevicePointer<T> ptr = Malloc(count);

    const cudaError_t error = cudaMemcpy(
        ptr.get(), source, count * sizeof(T), cudaMemcpyHostToDevice);
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);

    return ptr;
  }

  // FIXME: nope
  T *&get() noexcept { return ptr; }
  const T *get() const noexcept { return ptr; }
};

template <typename T> class PitchedView;
template <typename T> class ConstPitchedView;

template <typename T> class PitchedDevicePointer {
private:
  DevicePointer<T> ptr;
  size_t pitch;

  // FIXME
  PitchedDevicePointer(DevicePointer<T> &&ptr, size_t pitch)
      : ptr(std::move(ptr)), pitch(pitch) {}

public:
  PitchedDevicePointer(PitchedDevicePointer<T> &&other)
      : ptr(std::move(other.ptr)), pitch(other.pitch) {}

  PitchedDevicePointer(const PitchedDevicePointer<T> &other) = delete;
  PitchedDevicePointer<T>
  operator=(const PitchedDevicePointer<T> &other) = delete;
  PitchedDevicePointer<T> operator=(PitchedDevicePointer<T> &&other) = delete;

  static PitchedDevicePointer<T> MallocPitch(size_t width, size_t height) {
    T *ptr;
    size_t pitch;

    const cudaError_t error =
        cudaMallocPitch(&ptr, &pitch, width * sizeof(T), height);
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);

    return PitchedDevicePointer(DevicePointer(ptr), pitch);
  }

  static PitchedDevicePointer<T>
  MemcpyPitch(const T *source, size_t width, size_t height) {
    PitchedDevicePointer<T> ptr = MallocPitch(width, height);

    const cudaError_t error = cudaMemcpy2D(ptr.View().Ptr(),
                                           ptr.View().Pitch(),
                                           source,
                                           width * sizeof(T),
                                           width * sizeof(T),
                                           height,
                                           cudaMemcpyHostToDevice);
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);

    return ptr;
  }

  PitchedView<T> View() noexcept { return PitchedView<T>(ptr.get(), pitch); }

  ConstPitchedView<T> ConstView() const noexcept {
    return ConstPitchedView<T>(ptr.get(), pitch);
  }
};

template <typename T> class PitchedView {
private:
  T *ptr;
  size_t pitch;

public:
  __host__ __device__ PitchedView(T *ptr, size_t pitch)
      : ptr(ptr), pitch(pitch) {}

  __host__ __device__ T *operator[](size_t y) const noexcept {
    return ptr + y * (pitch / sizeof(T));
  }

  __host__ __device__ T *Ptr() { return ptr; }

  __host__ __device__ size_t Pitch() { return pitch; }

  template <typename U>
  __host__ __device__ operator PitchedView<U>() const noexcept {
    return PitchedView<U>(reinterpret_cast<U *>(ptr), pitch);
  }
};

template <typename T> class ConstPitchedView {
private:
  const T *ptr;
  size_t pitch;

public:
  __host__ __device__ ConstPitchedView(const T *ptr, size_t pitch)
      : ptr(ptr), pitch(pitch) {}

  __host__ __device__ const T *operator[](size_t y) const noexcept {
    return ptr + y * (pitch / sizeof(T));
  }

  __host__ __device__ const T *Ptr() { return ptr; }

  __host__ __device__ size_t Pitch() { return pitch; }

  template <typename U>
  __host__ __device__ operator ConstPitchedView<U>() const noexcept {
    return ConstPitchedView<U>(reinterpret_cast<const U *>(ptr), pitch);
  }
};

} // namespace caracal

#endif