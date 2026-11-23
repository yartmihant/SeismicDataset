// Instantiated in each CUDA solver's namespace with its own PET operators,
// precision, schedule and native launch ABI.
namespace SEMGPU2_PML_NAMESPACE {
namespace {
constexpr int pml_dim = SEMGPU2_PML_DIM;
__device__ inline int pml_voigt(int a, int b) {
    if (a == b) return a;
    return pml_dim == 2 ? 2 : 6-a-b;
}
__global__ void __launch_bounds__(THREADS_MAX) pml_response_kernel(ResponseLaunchData data, const SEMGPU2_PML_STATE* solution, const Real* sizes,
    const Real* mapping, const Real* density, Real* history, Real dt, Real* response) {
    constexpr int D = pml_dim, V = D*(D+1)/2;
    const auto slot = resolve_q_slot(data.slots_per_block,data.thread_slots,data.slot_entity_indices,data.slot_thread_offsets);
    const bool active = !slot.skip_thread;
    const Index elem = active ? data.field_entity_ids[slot.element_index] : 0;
    const auto nu = slot.local_q, start = slot.slot_start;
    const Index tid = threadIdx.x;
#if SEMGPU2_PML_HYBRID
    Code types[D]{};
    const int count = active ? load_element_pet_ids(elem,data.space.pet_ids,data.space.pet_offsets,types) : 0;
    const auto pos = active ? resolve_pq_local(data.space.pets,nu,types,count) : PQLocal{false,0,0};
    const bool has_p = active && pos.has_p;
    const Index local_p = pos.p_local;
#define PML_PET_ARGS types,count,
#else
    const bool has_p = active;
    const Index local_p = nu;
#define PML_PET_ARGS
#endif
    const Index q = active ? data.space.q_node_offsets[elem]+nu : 0;
    const Index src = active ? data.field_node_offsets[slot.element_index]+nu : 0;
    const Index n = data.n_count, stride = data.field_q_count;
    const Index node = has_p ? data.space.p_nodes[data.space.p_node_offsets[elem]+local_p] : 0;
    __shared__ Real work[D][D][THREADS_MAX];
    Real mid[D]{}, gradient[D][D]{}, u[D]{}, v[D]{};
    for (int c=0; c<D; ++c) {
        Real value = 0;
        if (has_p) value = solution[c*n+node] - Real(.5)*dt*solution[(D+c)*n+node]
                        + Real(.25)*dt*dt*solution[(2*D+c)*n+node];
        work[0][c][tid] = value;
    }
    __syncthreads();
    if (active) {
        Real reference[D][D]{};
        field::local_q_grad(data.space.pets,nu,start,work[0],PML_PET_ARGS reference);
        field::ref_to_phys_grad_q(q,mapping,data.space.q_count,reference,gradient);
        if constexpr (D == 3) field::local_q_value(data.space.pets,nu,start,work[0],PML_PET_ARGS mid);
    }
    __syncthreads();
    for (int c=0; c<D; ++c) {
        work[0][c][tid] = has_p ? solution[c*n+node] : 0;
        work[1][c][tid] = has_p ? solution[(D+c)*n+node] : 0;
    }
    __syncthreads();
    Real reference_flux[D][D]{}, inertia[D]{};
    if (active) {
        field::local_q_value(data.space.pets,nu,start,work[0],PML_PET_ARGS u);
        field::local_q_value(data.space.pets,nu,start,work[1],PML_PET_ARGS v);
        const Real* coeff = data.values+src;
        Real d[3]{}, H[D][D]{}, J[D][D]{};
        for (int l=0; l<D; ++l) {
            d[l] = coeff[l*stride];
            const Real a = coeff[(D+l)*stride], b = coeff[(2*D+l)*stride];
            for (int c=0; c<D; ++c) {
                const Index hrow = (l*D+c)*stride+src;
                const Real old = history[hrow];
                H[l][c] = a*old + b*gradient[c][l];
                history[hrow] = H[l][c];
                if constexpr (D == 3) {
                    J[l][c] = history[9*stride+hrow] + b*old + coeff[(9+l)*stride]*gradient[c][l];
                    history[9*stride+hrow] = J[l][c];
                }
            }
        }
        const Real first = d[0]+d[1]+d[2], second = d[0]*d[1]+d[0]*d[2]+d[1]*d[2];
        Real flux[D][D]{};
        for (int j=0; j<D; ++j) for (int l=0; l<D; ++l) {
            if (D == 2 && j != l) continue;
            for (int b=0; b<D; ++b) {
                Real value = (first-2*d[j])*H[l][b];
                if constexpr (D == 3) {
                    if (j == l) value += d[(j+1)%3]*d[(j+2)%3]*J[l][b];
                    else value = d[3-j-l]*(H[l][b]+d[l]*J[l][b]);
                }
                for (int c=0; c<D; ++c) {
                    const int x = pml_voigt(c,j), y = pml_voigt(b,l);
                    const int lo = x < y ? x : y, hi = x < y ? y : x;
                    const int row = (D == 2 ? 6 : 12)+lo*V-lo*(lo+1)/2+hi;
                    flux[j][c] += coeff[row*stride]*value;
                }
            }
        }
        for (int c=0; c<D; ++c) {
            inertia[c] = first*v[c]+second*u[c];
            if constexpr (D == 3) {
                const Index row = (18+c)*stride+src;
                history[row] += dt*mid[c];
                inertia[c] += d[0]*d[1]*d[2]*history[row];
            }
            inertia[c] *= sizes[q]*density[src];
            for (int j=0; j<D; ++j) flux[j][c] *= sizes[q];
        }
        field::phys_to_ref_div_q(q,mapping,data.space.q_count,flux,reference_flux);
    }
    __syncthreads();
    for (int j=0; j<D; ++j) for (int c=0; c<D; ++c) work[j][c][tid] = reference_flux[j][c];
    __syncthreads();
    Real force[1][D]{}, mass_force[D]{};
    if (has_p) field::local_p_div(data.space.pets,nu,start,work,PML_PET_ARGS force);
    __syncthreads();
    for (int c=0; c<D; ++c) work[0][c][tid] = inertia[c];
    __syncthreads();
    if (has_p) {
        field::local_p_value(data.space.pets,nu,start,work[0],PML_PET_ARGS mass_force);
        for (int c=0; c<D; ++c) atomicAdd(response+(2*D+c)*n+node,-force[0][c]-mass_force[c]);
    }
#undef PML_PET_ARGS
}
__global__ void check_pml_kernel(const Real* values, Index size, Code* flag) {
    const Index i = static_cast<Index>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<size && !isfinite(values[i])) atomicExch(flag,1);
}
}
cudaError_t assemble_pml_response(const ResponseLaunchData& data,const SEMGPU2_PML_STATE* u,const Real* w,const Real* jac,
                                  const Real* rho,Real* history,Real dt,Real* force,cudaStream_t stream) {
    if (!data.block_count) return cudaSuccess;
    if (!launchable(data.block_count)) return cudaErrorInvalidConfiguration;
    pml_response_kernel<<<static_cast<unsigned>(data.block_count),THREADS_MAX,0,stream>>>(data,u,w,jac,rho,history,dt,force);
    return cudaGetLastError();
}
cudaError_t check_pml_history(const Real* history,Index size,Code* flag,cudaStream_t stream) {
    if (!size) return cudaSuccess;
    const auto blocks = linear_blocks(size);
    if (!launchable(blocks)) return cudaErrorInvalidConfiguration;
    check_pml_kernel<<<static_cast<unsigned>(blocks),THREADS_PER_BLOCK,0,stream>>>(history,size,flag);
    return cudaGetLastError();
}
} // namespace SEMGPU2_PML_NAMESPACE
namespace SEMGPU2_PML_NAMESPACE {
void prepare_pml_field(PreparedField& field, const std::map<Code, PreparedField>& materials, double step) {
    auto law = [](int kind, const double* source, Index stride, Index q, const double* unit, double* result) {
        constexpr int D = SEMGPU2_PML_DIM, V = D*(D+1)/2;
        using E = SGSecondaryPhysicsTypeEnum;
        const int rows = kind < 2 ? 3 : (kind == 2 ? (D == 2 ? 5 : 10) : (D == 2 ? 7 : 22));
        Real parameters[22], strain[V], stress[V]{};
        for (int r=0; r<rows; ++r) parameters[r] = static_cast<Real>(source[r*stride+q]);
        for (int r=0; r<V; ++r) strain[r] = static_cast<Real>(unit[r]);
#if SEMGPU2_PML_DIM == 2
        switch (kind) {
        case 0: apply_constitutive_law_2d<E::MATERIAL_ISO_PLANE_STRAIN>(parameters,0,1,strain,stress); break;
        case 1: apply_constitutive_law_2d<E::MATERIAL_ISO_PLANE_STRESS>(parameters,0,1,strain,stress); break;
        case 2: apply_constitutive_law_2d<E::MATERIAL_ORTHOTROPIC>(parameters,0,1,strain,stress); break;
        case 3: apply_constitutive_law_2d<E::MATERIAL_ANISOTROPIC>(parameters,0,1,strain,stress); break;
        }
#else
        switch (kind) {
        case 0: apply_constitutive_law_3d<E::MATERIAL_ISOTROPIC>(parameters,0,1,strain,stress); break;
        case 2: apply_constitutive_law_3d<E::MATERIAL_ORTHOTROPIC>(parameters,0,1,strain,stress); break;
        case 3: apply_constitutive_law_3d<E::MATERIAL_ANISOTROPIC>(parameters,0,1,strain,stress); break;
        }
#endif
        for (int r=0; r<V; ++r) result[r] = stress[r];
    };
    semgpu2::common::prepare_pml<SEMGPU2_PML_DIM>(field,materials,step,law);
}
}
