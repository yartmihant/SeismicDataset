#pragma once

#include "data.hpp"
#include <cuda_runtime_api.h>
#include <exception>
#include <optional>
#include <utility>

namespace SEMGPU2_SEISMIC_NAMESPACE {
struct CudaError : std::runtime_error {
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};
inline void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        throw CudaError(std::string(operation) + ": " + cudaGetErrorString(code));
    }
}

// One error slot for RAII teardown; no allocator interception or byte counters.
struct CleanupStatus {
    cudaError_t error = cudaSuccess;
    void record(cudaError_t value) noexcept {
        if (error == cudaSuccess) error = value;
    }
};

template<class Function>
auto cuda_operation(Function function) {
    CleanupStatus cleanup;
    using Result = decltype(function(cleanup));
    std::optional<Result> result;
    std::exception_ptr failure;
    try {
        result.emplace(function(cleanup));
    } catch (...) {
        failure = std::current_exception();
    }
    // Lambda locals have already released their resources, including on error.
    cuda_check(cleanup.error, "CUDA resource cleanup");
    if (failure) std::rethrow_exception(failure);
    return std::move(*result);
}

class CudaStream {
    CleanupStatus& cleanup_;
    cudaStream_t stream_ = nullptr;
public:
    explicit CudaStream(CleanupStatus& cleanup) : cleanup_(cleanup) {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~CudaStream() {
        if (stream_) {
            cleanup_.record(cudaStreamSynchronize(stream_));
            cleanup_.record(cudaStreamDestroy(stream_));
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
    void synchronize() const { cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
};

template<class T>
class DeviceBuffer {
    CleanupStatus& cleanup_;
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
public:
    DeviceBuffer(CleanupStatus& cleanup, Index count) : cleanup_(cleanup), count_(array_size<T>(count)) {
        if (count_) cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count_ * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (pointer_) cleanup_.record(cudaFree(pointer_)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    Index size() const { return static_cast<Index>(count_); }
    void upload(const T* source, cudaStream_t stream) {
        if (count_) cuda_check(cudaMemcpyAsync(pointer_, source, count_ * sizeof(T), cudaMemcpyHostToDevice, stream), "H2D");
    }
    void download(T* target, cudaStream_t stream) const {
        if (count_) cuda_check(cudaMemcpyAsync(target, pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream), "D2H");
    }
    void clear(cudaStream_t stream) {
        if (count_) cuda_check(cudaMemsetAsync(pointer_, 0, count_ * sizeof(T), stream), "cudaMemsetAsync");
    }
};
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
