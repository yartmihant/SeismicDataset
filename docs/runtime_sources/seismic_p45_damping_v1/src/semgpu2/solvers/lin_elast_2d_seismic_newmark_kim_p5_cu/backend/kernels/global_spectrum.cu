#include "../spectrum.hpp"
#include <cmath>
#include <limits>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
__device__ inline Real reduce_block(Real value, Real* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = THREADS_PER_BLOCK / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    return shared[0];
}
__global__ void initialize_kernel(Real* current, Real* previous, Real* partials, Index n_count) {
    Real total = 0.0;
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * n_count; i += static_cast<Index>(gridDim.x) * blockDim.x) {
        std::uint32_t seed = static_cast<std::uint32_t>(i + 1);
        seed = ((seed ^ (seed >> 16)) * 0x45D9F3Bu); seed = ((seed ^ (seed >> 16)) * 0x45D9F3Bu); seed ^= seed >> 16;
        const Real value = static_cast<Real>(seed) / 2147483648.0 - 1.0;
        current[i] = value; previous[i] = 0.0; total += value * value;
    }
    __shared__ Real shared[THREADS_PER_BLOCK];
    const Real sum = reduce_block(total, shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}
__global__ void load_probe_kernel(const Real* current, const Real* mass, Real* solution, Code row, Index n) {
    const Index i=static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<2*n) { const Index c=i/n,node=i%n; solution[(row+c)*n+node]=current[i]/sqrt(mass[node]); }
        }
__global__ void action_kernel(const Real* response, const Real* mass, const Real* current, const Real* previous, Real* residual, Real old_beta, Code row, Real* partials, Index n) {
    Real total = 0.0;
    // Scalar contributions to the global dot product retain grid-stride reduction.
    for(Index i=static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x;i<2*n;i+=static_cast<Index>(gridDim.x)*blockDim.x) {
        const Index c=i/n,node=i%n;
        const Real value=-response[(row+c)*n+node]/sqrt(mass[node])-old_beta*previous[i];
        residual[i]=value; total+=current[i]*value;
    }
    __shared__ Real shared[THREADS_PER_BLOCK];
    const Real sum = reduce_block(total, shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}
__global__ void subtract_kernel(Real* residual, const Real* current, Real alpha, Real* partials, Index count) {
    Real total = 0.0;
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += static_cast<Index>(gridDim.x) * blockDim.x) {
        const Real value = residual[i] - alpha * current[i]; residual[i] = value; total += value * value;
    }
    __shared__ Real shared[THREADS_PER_BLOCK];
    const Real sum = reduce_block(total, shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}
__global__ void advance_kernel(Real* previous, Real* current, const Real* residual, Real beta, Index count) {
    for (Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += static_cast<Index>(gridDim.x) * blockDim.x) { previous[i] = current[i]; current[i] = residual[i] / beta; }
}
__global__ void sum_kernel(const Real* partials, Index count, Real* result) {
    Real total = 0.0; for (Index i = threadIdx.x; i < count; i += blockDim.x) total += partials[i];
    __shared__ Real shared[THREADS_PER_BLOCK];
    const Real sum = reduce_block(total, shared);
    if (threadIdx.x == 0) result[0] = sum;
}
inline cudaError_t blocks_ok(Index blocks) { return blocks > 0 && blocks <= std::numeric_limits<int>::max() ? cudaSuccess : cudaErrorInvalidConfiguration; }
}
cudaError_t initialize_lanczos_vectors(Real* c, Real* p, Real* sums, Index n, Index b, cudaStream_t s) { if (auto e=blocks_ok(b);e!=cudaSuccess)return e; initialize_kernel<<<(unsigned)b,THREADS_PER_BLOCK,0,s>>>(c,p,sums,n);return cudaGetLastError(); }
cudaError_t load_lanczos_probe(const Real* c,const Real*m,Real*u,Code r,Index n,Index b,cudaStream_t s){if(auto e=blocks_ok(b);e!=cudaSuccess)return e;load_probe_kernel<<<static_cast<unsigned>((2*n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK),THREADS_PER_BLOCK,0,s>>>(c,m,u,r,n);return cudaGetLastError();}
cudaError_t assemble_lanczos_action(const Real*r,const Real*m,const Real*c,const Real*p,Real*x,Real bta,Code row,Real*sums,Index n,Index b,cudaStream_t s){if(auto e=blocks_ok(b);e!=cudaSuccess)return e;action_kernel<<<(unsigned)b,THREADS_PER_BLOCK,0,s>>>(r,m,c,p,x,bta,row,sums,n);return cudaGetLastError();}
cudaError_t subtract_lanczos_projection(Real*x,const Real*c,Real a,Real*sums,Index count,Index b,cudaStream_t s){if(auto e=blocks_ok(b);e!=cudaSuccess)return e;subtract_kernel<<<(unsigned)b,THREADS_PER_BLOCK,0,s>>>(x,c,a,sums,count);return cudaGetLastError();}
cudaError_t advance_lanczos_vectors(Real*p,Real*c,const Real*x,Real bta,Index count,Index b,cudaStream_t s){if(auto e=blocks_ok(b);e!=cudaSuccess)return e;advance_kernel<<<(unsigned)b,THREADS_PER_BLOCK,0,s>>>(p,c,x,bta,count);return cudaGetLastError();}
cudaError_t sum_lanczos_partials(const Real*sums,Index count,Real*result,cudaStream_t s){sum_kernel<<<1,THREADS_PER_BLOCK,0,s>>>(sums,count,result);return cudaGetLastError();}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
