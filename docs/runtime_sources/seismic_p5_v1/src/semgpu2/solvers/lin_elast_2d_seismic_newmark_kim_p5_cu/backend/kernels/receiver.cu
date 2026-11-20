#include "../receivers.hpp"
#include <cmath>
#include <limits>

namespace semgpu2_seismic_p5_cu {
    namespace {
        __global__ void velocity_frame_kernel(VelocityFrame data, Index count) {
            const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) return;
            const Index c = i / data.entries, j = i % data.entries;
            const Index row = data.components[c] == static_cast<Code>(SGReceiverComponentEnum::VX)
                ? static_cast<Index>(SGNSolutionComponentEnum::VX) : static_cast<Index>(SGNSolutionComponentEnum::VY);
            atomicAdd(data.frame + c * data.frame_size + data.destinations[j],
                data.weights[j] * data.solution[row * data.n_count + data.nodes[j]]);
        }
        __global__ void flag_observation_kernel(const Real* values, Index count, Code* nonfinite) {
            const Index i = static_cast<Index>(blockIdx.x)*blockDim.x + threadIdx.x;
            if (i < count && !isfinite(values[i])) atomicExch(nonfinite, static_cast<Code>(1));
        }
        cudaError_t linear_blocks(Index count, int threads, int& blocks) {
            if (count < 0 || threads <= 0) return cudaErrorInvalidValue;
            if (!count) {
                blocks = 0;
                return cudaSuccess;
            }
            const Index result = (count - 1) / threads + 1;
            if (result > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
            blocks = static_cast<int>(result);
            return cudaSuccess;
        }
    }
    // namespace
    cudaError_t flag_receiver_frame(const Real* frame, Index size, Code* flag, cudaStream_t stream) {
        int blocks = 0;
        if (const auto status = linear_blocks(size, THREADS_PER_BLOCK, blocks); status != cudaSuccess) return status;
        if (!blocks) return cudaSuccess;
        flag_observation_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(frame, size, flag);
        return cudaGetLastError();
    }
    cudaError_t gather_velocity_frame(const VelocityFrame& data, cudaStream_t stream) {
        const Index count = checked_product(data.entries, data.component_count);
        int blocks = 0;
        if (auto status = linear_blocks(count, THREADS_PER_BLOCK, blocks); status != cudaSuccess) return status;
        if (!blocks) return cudaSuccess;
        velocity_frame_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(data, count);
        return cudaGetLastError();
    }
}
