#include "../stepping.hpp"
#include <cmath>
#include <limits>
namespace SEMGPU2_SEISMIC_NAMESPACE {
    namespace {
        constexpr int D = 2;
        inline bool linear_launchable(Index count) {
            return count >= 0 && (count == 0 || (count - 1) / THREADS_PER_BLOCK + 1 <= std::numeric_limits<int>::max());
        }
        inline Index linear_blocks(Index count) {
            return count == 0 ? 0 : (count - 1) / THREADS_PER_BLOCK + 1;
        }
        __global__ void predict_nodal_state_kernel(Real* u, Index n, Real dt) {
            const Index node = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x, c = blockIdx.y, shot = blockIdx.z;
            if (node >= n) return;
            const Index base = shot * 3 * D * n;
            const Real a = u[base + (2 * D + c) * n + node];
            u[base + c * n + node] += dt * u[base + (D + c) * n + node] + 0.5 * dt * dt * a;
            u[base + (D + c) * n + node] += dt * a;
        }
        __global__ void correct_nodal_state_kernel(Real* u, const Real* r, const Real* mass, Index n, Real dt, Real beta, Real gamma) {
            const Index node = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x, c = blockIdx.y, shot = blockIdx.z;
            if (node >= n) return;
            const Index u_base = shot * 3 * D * n, r_base = shot * D * n;
            const Real a = r[r_base + c * n + node] / mass[node];
            const Real delta = a - u[u_base + (2 * D + c) * n + node];
            u[u_base + c * n + node] += beta * dt * dt * delta;
            u[u_base + (D + c) * n + node] += gamma * dt * delta;
            u[u_base + (2 * D + c) * n + node] = a;
        }
        __global__ void finite_kernel(const Real* u, const Real* r, Index n, Index shots, Code* flag) {
            const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < shots * 3 * D * n && !isfinite(u[i])) atomicExch(flag, static_cast<Code>(1));
            if (i < shots * D * n && !isfinite(r[i])) atomicExch(flag, static_cast<Code>(1));
        }
        __global__ void finite_values_kernel(const Real* values, Index count, Code* flag) {
            const Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index < count && !isfinite(values[index])) atomicExch(flag, static_cast<Code>(1));
        }
    }

    cudaError_t predict_nodal_state(Real* u, Index n, Real dt, Index shots, cudaStream_t s) {
        if (!n) return cudaSuccess;
        if (!linear_launchable(n)) return cudaErrorInvalidConfiguration;
        predict_nodal_state_kernel<<<dim3(static_cast<unsigned>(linear_blocks(n)), 2, static_cast<unsigned>(shots)),THREADS_PER_BLOCK,0,s>>>(u, n, dt);
        return cudaGetLastError();
    }
    cudaError_t correct_nodal_state(Real* u, Real* r, const Real* m, Index n, Real dt, Real b, Real g, Index shots, cudaStream_t s) {
        if (!n) return cudaSuccess;
        if (!linear_launchable(n)) return cudaErrorInvalidConfiguration;
        correct_nodal_state_kernel<<<dim3(static_cast<unsigned>(linear_blocks(n)), 2, static_cast<unsigned>(shots)),THREADS_PER_BLOCK,0,s>>>(u, r, m, n, dt, b, g);
        return cudaGetLastError();
    }
    cudaError_t flag_nonfinite_solution(const Real* u, const Real* r, Index n, Index shots, Code* f, cudaStream_t s) {
        const Index total = checked_product(checked_product(3 * D, n), shots);
        if (!total) return cudaSuccess;
        if (!linear_launchable(total)) return cudaErrorInvalidConfiguration;
        finite_kernel<<<static_cast<unsigned>(linear_blocks(total)),THREADS_PER_BLOCK,0,s>>>(u, r, n, shots, f);
        return cudaGetLastError();
    }
    cudaError_t flag_nonfinite_values(const Real* values, Index count, Code* flag, cudaStream_t stream) {
        if (!count) return cudaSuccess;
        if (!linear_launchable(count)) return cudaErrorInvalidConfiguration;
        finite_values_kernel<<<static_cast<unsigned>(linear_blocks(count)), THREADS_PER_BLOCK, 0, stream>>>(values, count, flag);
        return cudaGetLastError();
    }
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
