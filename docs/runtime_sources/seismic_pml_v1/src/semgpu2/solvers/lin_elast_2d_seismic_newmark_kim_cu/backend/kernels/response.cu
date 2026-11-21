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
            const auto point = locate_element_point(data.entity_count);
            const bool active = point.active;
            const Index element = point.element;
            const Index q = active ? element * data.space.nodes_per_element + point.local_q : 0;
            const Index node = active ? data.space.p_nodes[q] : 0;
            const Index count = data.space.q_count;
            __shared__ Real displacement[2][THREADS_MAX];
            const int thread_index = point.element_start + point.local_q;
            displacement[0][thread_index] = active ? solution[node] : 0;
            displacement[1][thread_index] = active ? solution[data.n_count + node] : 0;
            Real strain[3], stress[3]{};
            compute_strain_q_2d(active, data.space.pets, point.local_q, point.element_start,
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
            integrate_stress_p_2d(active, data.space.pets, point.local_q, point.element_start,
                stress, q, sizes, difgeoms, count, workspace, force);
            if (active) {
                atomicAdd(response + node, -force[0][0]);
                atomicAdd(response + data.n_count + node, -force[0][1]);
            }
        }
        // The PML field is stored compactly over its selected MAIN elements.
        // `history` holds Hx_x, Hx_y, Hy_x and Hy_y in four contiguous rows.
        // Hx and Hy solve Hdot + d H = grad_x(u) and grad_y(u), respectively.
        // The update uses the exact constant-gradient propagator with the
        // midpoint gradient reconstructed from the Newmark predictor.
        __global__ void pml_response_kernel(ResponseLaunchData data, const Real* solution,
            const Real* materials, const Real* sizes, const Real* difgeoms,
            Real* history, Real dt, Real* response) {
            const auto point = locate_element_point(data.entity_count);
            const bool active = point.active;
            const Index compact = active ? point.element * data.space.nodes_per_element + point.local_q : 0;
            const Index element = active ? data.field_entity_ids[point.element] : 0;
            const Index q = active ? element * data.space.nodes_per_element + point.local_q : 0;
            const Index node = active ? data.space.p_nodes[q] : 0;
            const Index q_count = data.space.q_count;
            const int tid = point.element_start + point.local_q;

            __shared__ Real midpoint_u[2][THREADS_MAX];
            if (active) {
                // Predictor has overwritten U and V, while A remains a_n.
                for (int component = 0; component < 2; ++component) {
                    const Real u = solution[component * data.n_count + node];
                    const Real v = solution[(2 + component) * data.n_count + node];
                    const Real a = solution[(4 + component) * data.n_count + node];
                    const Real previous = u - dt * v + 0.5 * dt * dt * a;
                    midpoint_u[component][tid] = 0.5 * (u + previous);
                }
            } else {
                midpoint_u[0][tid] = midpoint_u[1][tid] = 0.0;
            }

            Real midpoint_ref_grad[2][2]{};
            Real midpoint_grad[2][2]{};
            __syncthreads();
            if (active) {
                field::local_q_grad<2, THREADS_MAX, 2>(data.space.pets, point.local_q,
                    point.element_start, midpoint_u, midpoint_ref_grad);
                field::ref_to_phys_grad_q<2, 2, 2>(q, difgeoms, q_count,
                    midpoint_ref_grad, midpoint_grad);
            }

            Real physical_flux[2][2]{};
            Real dx = 0.0;
            Real dy = 0.0;
            if (active) {
                dx = data.values[compact];
                dy = data.values[data.field_q_count + compact];
                const Real ax = exp(-dx * dt);
                const Real ay = exp(-dy * dt);
                const Real bx = dx == 0.0 ? dt : -expm1(-dx * dt) / dx;
                const Real by = dy == 0.0 ? dt : -expm1(-dy * dt) / dy;
                const Real midpoint_x[2]{
                    midpoint_grad[0][0], midpoint_grad[1][0]};
                const Real midpoint_y[2]{
                    midpoint_grad[0][1], midpoint_grad[1][1]};
                Real hx[2];
                Real hy[2];
                for (int component = 0; component < 2; ++component) {
                    hx[component] = ax * history[component * data.field_q_count + compact] + bx * midpoint_x[component];
                    hy[component] = ay * history[(2 + component) * data.field_q_count + compact] + by * midpoint_y[component];
                    history[component * data.field_q_count + compact] = hx[component];
                    history[(2 + component) * data.field_q_count + compact] = hy[component];
                }
                const Real lambda = materials[q_count + q];
                const Real mu = materials[2 * q_count + q];
                // Bx^T Axx Hx and By^T Ayy Hy are the two directional
                // stiffness corrections. Cross-derivative terms stay in the
                // ordinary elastic action and therefore do not appear here.
                physical_flux[0][0] = (dy - dx) * (lambda + 2.0 * mu) * hx[0];
                physical_flux[0][1] = (dy - dx) * mu * hx[1];
                physical_flux[1][0] = (dx - dy) * mu * hy[0];
                physical_flux[1][1] = (dx - dy) * (lambda + 2.0 * mu) * hy[1];
            }
            __shared__ Real workspace[2][2][THREADS_MAX];
            Real force[1][2]{};
            if (active) {
                Real weighted_flux[2][2];
                for (int direction = 0; direction < 2; ++direction)
                    for (int component = 0; component < 2; ++component)
                        weighted_flux[direction][component] = physical_flux[direction][component] * sizes[q];
                Real reference_flux[2][2];
                field::phys_to_ref_div_q<2, 2, 2>(q, difgeoms, q_count,
                    weighted_flux, reference_flux);
                for (int reference = 0; reference < 2; ++reference)
                    for (int component = 0; component < 2; ++component)
                        workspace[reference][component][tid] = reference_flux[reference][component];
            } else {
                for (int reference = 0; reference < 2; ++reference)
                    for (int component = 0; component < 2; ++component)
                        workspace[reference][component][tid] = 0.0;
            }
            __syncthreads();
            if (active) {
                field::local_p_div<2, THREADS_MAX, 2>(data.space.pets, point.local_q,
                    point.element_start, workspace, force);
                const Real density = materials[q];
                const Real mass = density * sizes[q];
                const Real inertia_x = mass * ((dx + dy) * solution[2 * data.n_count + node] + dx * dy * solution[node]);
                const Real inertia_y = mass * ((dx + dy) * solution[3 * data.n_count + node] + dx * dy * solution[data.n_count + node]);
                atomicAdd(response + node, -force[0][0] - inertia_x);
                atomicAdd(response + data.n_count + node, -force[0][1] - inertia_y);
            }
        }
        __global__ void boundary_damping_kernel(ResponseLaunchData data, const Real* solution, const Real* sizes, const Real* normals, Real* response) {
            const auto point = locate_element_point(data.entity_count);
            if (!point.active) return;
            const Index element = data.field_entity_ids[point.element];

            const Index qid = element * data.space.nodes_per_element + point.local_q;
            const Index node = data.space.p_nodes[qid];
            const Real vx = solution[2 * data.n_count + node], vy = solution[3 * data.n_count + node];
            const Index source = point.element * data.space.nodes_per_element + point.local_q;
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
        const dim3 block = element_block(data.space.nodes_per_element);
        const Index blocks = element_block_count(data.entity_count, block);
        if (!blocks) return cudaSuccess;
        if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
        material_response_kernel<<<static_cast<unsigned>(blocks), block, 0, stream>>>(
            data, solution, sizes, difgeoms, response);
        return cudaGetLastError();
    }
    cudaError_t assemble_pml_response(const ResponseLaunchData& data, const Real* solution,
        const Real* materials, const Real* sizes, const Real* difgeoms, Real* history,
        Real dt, Real* response, cudaStream_t stream) {
        const dim3 block = element_block(data.space.nodes_per_element);
        const Index blocks = element_block_count(data.entity_count, block);
        if (!blocks) return cudaSuccess;
        if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
        pml_response_kernel<<<static_cast<unsigned>(blocks), block, 0, stream>>>(
            data, solution, materials, sizes, difgeoms, history, dt, response);
        return cudaGetLastError();
    }
    cudaError_t assemble_boundary_damping(const ResponseLaunchData& d, const Real* u, const Real* s, const Real* n, Real* r, cudaStream_t stream) {
        const dim3 block = element_block(d.space.nodes_per_element);
        const Index blocks = element_block_count(d.entity_count, block);
        if (!blocks) return cudaSuccess;
        if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
        boundary_damping_kernel<<<static_cast<unsigned>(blocks), block, 0, stream>>>(d,u,s,n,r);
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
