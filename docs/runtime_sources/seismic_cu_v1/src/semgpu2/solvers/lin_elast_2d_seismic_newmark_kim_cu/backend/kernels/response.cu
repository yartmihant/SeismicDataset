#include "../stepping.hpp"
#include "../operators/material.cuh"
#include <limits>
namespace semgpu2_seismic_cu {
    namespace field = semgpu2::common::cu::tenzor;
    namespace {
        inline bool launchable(Index blocks) {
            return blocks >= 0 && blocks <= std::numeric_limits<int>::max();
        }
        inline Index linear_blocks(Index count) {
            return count == 0 ? 0 : (count - 1) / THREADS_PER_BLOCK + 1;
        }
        __global__ void material_response_kernel(ResponseLaunchData data, const Real* solution,
            const Real* sizes, const Real* difgeoms, Real* response) {
            const auto slot = resolve_q_slot(data.slots_per_block, data.thread_slots,
                data.slot_entity_indices, data.slot_thread_offsets);
            const bool active = !slot.skip_thread;
            const Index element = slot.element_index;
            const Index node = active ? data.space.p_nodes[data.space.p_node_offsets[element] + slot.local_q] : 0;
            const Index q = active ? data.space.q_node_offsets[element] + slot.local_q : 0;
            const Index count = data.space.q_count;
            __shared__ Real displacement[2][THREADS_MAX];
            displacement[0][slot.thread_index] = active ? solution[node] : 0;
            displacement[1][slot.thread_index] = active ? solution[data.n_count + node] : 0;
            Real strain[3], stress[3]{};
            compute_strain_q_2d(active, data.space.pets, slot.local_q, slot.slot_start,
                displacement, q, difgeoms, count, strain);
            if (active) {
                const Real lambda = data.values[count + q];
                const Real mu = data.values[2 * count + q];
                const Real pressure = lambda * (strain[0] + strain[1]);
                stress[0] = pressure + 2 * mu * strain[0];
                stress[1] = pressure + 2 * mu * strain[1];
                stress[2] = mu * strain[2];
            }
            __shared__ Real workspace[2][2][THREADS_MAX];
            Real force[1][2];
            integrate_stress_p_2d(active, data.space.pets, slot.local_q, slot.slot_start,
                stress, q, sizes, difgeoms, count, workspace, force);
            if (active) {
                atomicAdd(response + node, -force[0][0]);
                atomicAdd(response + data.n_count + node, -force[0][1]);
            }
        }
        __global__ void boundary_damping_kernel(ResponseLaunchData data, const Real* solution, const Real* sizes, const Real* normals, Real* response) {
            const auto slot = resolve_q_slot(data.slots_per_block, data.thread_slots, data.slot_entity_indices, data.slot_thread_offsets);
            if (slot.skip_thread) return;
            const Index element = data.field_entity_ids[slot.element_index];

            const Index node = data.space.p_nodes[data.space.p_node_offsets[element] + slot.local_q];
            const Real vx = solution[2 * data.n_count + node], vy = solution[3 * data.n_count + node];
            const Index qid = data.space.q_node_offsets[element] + slot.local_q, source = data.field_node_offsets[slot.element_index] + slot.local_q;
            // normals is the two-row NX/NY view, not the full boundary geometry.
            const Real nx = normals[qid], ny = normals[data.space.q_count + qid], tx = -ny, ty = nx, vn = vx * nx + vy * ny, vt = vx * tx + vy * ty;
            Real force[2]{
                -data.values[source] * vn * nx - data.values[data.field_q_count + source] * vt * tx, -data.values[source] * vn * ny - data.values[data.field_q_count + source] * vt * ty
            };
            field::apply_q_measure(qid, sizes, force);
            atomicAdd(response + node, force[0]);
            atomicAdd(response + data.n_count + node, force[1]);
        }
        __global__ void nodal_force_kernel(const Index* ids, const Real* values, Index count, Index n, Real* response) {
            const Index p=static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x,c=blockIdx.y;
            if(p>=count) return;
            const Index node=ids[p];
            atomicAdd(response+c*n+node,values[c*count+p]);
        }
        __global__ void nodal_damping_kernel(const Index* ids, const Real* values, Index count, Index n, const Real* mass, const Real* solution, Real* response) {
            const Index p=static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x,c=blockIdx.y;
            if(p>=count) return;
            const Index node=ids[p]; const Real coefficient=values[p]*mass[node];
            atomicAdd(response+c*n+node,-coefficient*solution[(2+c)*n+node]);
        }
    }
    cudaError_t assemble_material_response(const ResponseLaunchData& data, const Real* solution,
        const Real* sizes, const Real* difgeoms, Real* response, cudaStream_t stream) {
        if (!data.block_count) return cudaSuccess;
        if (!launchable(data.block_count)) return cudaErrorInvalidConfiguration;
        material_response_kernel<<<static_cast<unsigned>(data.block_count), THREADS_MAX, 0, stream>>>(
            data, solution, sizes, difgeoms, response);
        return cudaGetLastError();
    }
    cudaError_t assemble_boundary_damping(const ResponseLaunchData& d, const Real* u, const Real* s, const Real* n, Real* r, cudaStream_t stream) {
        if (!d.block_count) return cudaSuccess;
        if (!launchable(d.block_count)) return cudaErrorInvalidConfiguration;
        boundary_damping_kernel<<<static_cast<unsigned>(d.block_count), THREADS_MAX, 0, stream>>>(d,u,s,n,r);
        return cudaGetLastError();
    }
    cudaError_t assemble_nodal_force(const DevicePreparedField& f, const Real* v, Index n, Real* r, cudaStream_t stream) {
        const Index count = static_cast<Index>(f.host->entity_ids.size());
        const Index blocks = linear_blocks(count);
        if (!count) return cudaSuccess;
        if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
        nodal_force_kernel<<<dim3(static_cast<unsigned>(blocks), 2),THREADS_PER_BLOCK,0,stream>>>(f.entity_ids.get(),v,count,n,r);
        return cudaGetLastError();
    }
    cudaError_t assemble_nodal_damping(const DevicePreparedField& f, const Real* v, Index n, const Real* m, const Real* u, Real* r, cudaStream_t stream) {
        const Index count = static_cast<Index>(f.host->entity_ids.size());
        const Index blocks = linear_blocks(count);
        if (!count) return cudaSuccess;
        if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
        nodal_damping_kernel<<<dim3(static_cast<unsigned>(blocks), 2),THREADS_PER_BLOCK,0,stream>>>(f.entity_ids.get(),v,count,n,m,u,r);
        return cudaGetLastError();
    }
}
// namespace semgpu2_seismic_cu
