#include <preparation.hpp>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
// In a collocated PET every MAIN Q position has exactly one P-node image.
__global__ void mass_kernel(const Index* nodes, const Real* density, const Real* sizes,
    Index count, Real* mass) {
    const Index q = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q < count) atomicAdd(mass + nodes[q], density[q] * sizes[q]);
}
}
cudaError_t assemble_mass_2d(const Index* nodes, const Real* density, const Real* sizes,
    Index count, Real* mass, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    const Index blocks = (count - 1) / THREADS_PER_BLOCK + 1;
    if (blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    mass_kernel<<<static_cast<unsigned>(blocks), THREADS_PER_BLOCK, 0, stream>>>(nodes, density, sizes, count, mass);
    return cudaGetLastError();
}
}
