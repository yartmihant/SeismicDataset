#include "data.hpp"
#include "device_data.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace semgpu2_seismic_cu {
namespace {
void require_cuda_device() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw CudaError("A visible CUDA device is required");
    // Initialize the current device context without retaining an allocation.
    cuda_check(cudaFree(nullptr), "initialize CUDA device context");
}

void offsets(const std::vector<Index>& values, std::size_t count, std::size_t total) {
    if (values.size() != count + 1 || values.front() != 0
        || values.back() < 0 || static_cast<std::uint64_t>(values.back()) != total) {
        throw std::invalid_argument("Invalid packed array offsets");
    }
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] < values[i - 1]) throw std::invalid_argument("Offsets must be nondecreasing");
    }
}

void finite(const std::vector<Real>& values) {
    for (Real value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite PET coefficient");
    }
}

void validate_pets(const PETData& pets) {
    const auto count = pets.p_node_counts.size();
    if (count != 1 || pets.q_node_counts.size() != count || pets.dims.size() != count
        || pets.orb_counts.size() != count || pets.symm_counts.size() != count) {
        throw std::invalid_argument("Inconsistent PET catalog columns");
    }
    offsets(pets.q_weight_offsets, count, pets.q_weights.size());
    offsets(pets.pq_value_offsets, count, pets.pq_values.size());
    offsets(pets.pq_grad_offsets, count, pets.pq_grads.size());
    offsets(pets.q_node_orb_offsets, count, pets.q_node_orbs.size());
    offsets(pets.q_node_rep_symm_offsets, count, pets.q_node_rep_symms.size());
    offsets(pets.symm_grad_op_offsets, count, pets.symm_grad_ops.size());
    offsets(pets.p_symm_inv_node_offsets, count, pets.p_symm_inv_nodes.size());
    finite(pets.q_weights); finite(pets.pq_values); finite(pets.pq_grads); finite(pets.symm_grad_ops);
    for (std::size_t id = 0; id < count; ++id) {
        const Index p = pets.p_node_counts[id], q = pets.q_node_counts[id];
        const Index o = pets.orb_counts[id], s = pets.symm_counts[id];
        const Code dim = pets.dims[id];
        if (p <= 0 || q != p || o != q || s != 1 || dim != 1
            || pets.q_weight_offsets[id + 1] - pets.q_weight_offsets[id] != o
            || pets.pq_value_offsets[id + 1] - pets.pq_value_offsets[id] != checked_product(o, p)
            || pets.pq_grad_offsets[id + 1] - pets.pq_grad_offsets[id] != checked_product(dim, checked_product(o, p))
            || pets.q_node_orb_offsets[id + 1] - pets.q_node_orb_offsets[id] != q
            || pets.q_node_rep_symm_offsets[id + 1] - pets.q_node_rep_symm_offsets[id] != q
            || pets.symm_grad_op_offsets[id + 1] - pets.symm_grad_op_offsets[id] != checked_product(s, checked_product(dim, dim))
            || pets.p_symm_inv_node_offsets[id + 1] - pets.p_symm_inv_node_offsets[id] != checked_product(s, p)) {
            throw std::invalid_argument("Invalid compressed PET shape");
        }
        std::vector<bool> seen(array_size<bool>(q), false);
        for (Index r = 0; r < q; ++r) {
            const Index orb = pets.q_node_orbs[pets.q_node_orb_offsets[id] + r];
            if (orb < 0 || orb >= q || seen[orb]
                || pets.q_node_rep_symms[pets.q_node_rep_symm_offsets[id] + r] != 0
                || pets.p_symm_inv_nodes[pets.p_symm_inv_node_offsets[id] + r] != r)
                throw std::invalid_argument("tenzor PET requires singleton Q orbits and identity P permutation");
            seen[orb] = true;
            for (Index i = 0; i < p; ++i) {
                if (std::abs(pets.pq_values[pets.pq_value_offsets[id] + orb * p + i] - (r == i ? 1.0 : 0.0)) > 1e-12)
                    throw std::invalid_argument("tenzor PET value operator must be identity within 1e-12");
            }
        }
        for (Code a = 0; a < dim; ++a)
            for (Code b = 0; b < dim; ++b)
                if (std::abs(pets.symm_grad_ops[pets.symm_grad_op_offsets[id] + a * dim + b] - (a == b ? 1.0 : 0.0)) > 1e-12)
                    throw std::invalid_argument("tenzor PET gradient transform must be identity within 1e-12");
    }
}

void validate_space(const SpaceData& space, const PETData& pets, Index n_count, int rank) {
    if (space.dim != 2 || space.degree != rank || space.rank != rank || space.pet_offsets.empty()) {
        throw std::invalid_argument("Unsupported PQ-space signature");
    }
    const auto count = space.pet_offsets.size() - 1;
    if (rank == 2 && count == 0) throw std::invalid_argument("MAIN requires an element");
    offsets(space.pet_offsets, count, space.pet_ids.size());
    offsets(space.p_node_offsets, count, space.p_nodes.size());
    if (space.q_node_offsets.size() != count + 1 || space.q_node_offsets.front() != 0) {
        throw std::invalid_argument("Invalid Q offsets");
    }
    semgpu2::common::cu::validate_element_q_capacity(space.q_node_offsets, THREADS_MAX);
    for (std::size_t element = 0; element < count; ++element) {
        const Index first = space.pet_offsets[element], last = space.pet_offsets[element + 1];
        if (last - first != rank) throw std::invalid_argument("Unsupported PET factor count");
        Index p = 1;
        for (Index k = first; k < last; ++k) {
            const Code id = space.pet_ids[k];
            if (id != 0) throw std::invalid_argument("PET ID outside catalog");
            p = checked_product(p, pets.p_node_counts[0]);
        }
        if (p != space.p_node_offsets[element + 1] - space.p_node_offsets[element]
            || p != space.q_node_offsets[element + 1] - space.q_node_offsets[element]) {
            throw std::invalid_argument("PQ offsets do not match PET factors");
        }
    }
    for (Index node : space.p_nodes) {
        if (node < 0 || node >= n_count) throw std::invalid_argument("P-node mapping outside N");
    }
}

TenzorPETData normalize_pets(const PETData& pets) {
    TenzorPETData result;
    const Index n = pets.p_node_counts[0];
    result.node_count = n;
    result.q_weights.resize(array_size<Real>(n));
    result.pq_grads.resize(array_size<Real>(checked_product(n, n)));
    for (Index q = 0; q < n; ++q) {
        const Index orbit = pets.q_node_orbs[q];
        result.q_weights[q] = pets.q_weights[orbit];
        for (Index p = 0; p < n; ++p)
            result.pq_grads[q * n + p] = pets.pq_grads[orbit * n + p];
    }
    return result;
}

template<class Enum> constexpr Index rows(Enum last) { return static_cast<Index>(last) + 1; }
void allocate_fields(CompiledState& state) {
    const Index n = state.n_count, main = state.main.q_count(), boundary = state.boundary.q_count();
    state.solution = {Field(rows(SGNSolutionComponentEnum::AY), n), Field(0, main), Field(0, boundary)};
    state.geometry = {Field(rows(SGNGeometryComponentEnum::SIZE), n),
        Field(rows(SGQMainGeometryComponentEnum::DIFGEOMETRY_YY), main),
        Field(rows(SGQBoundaryGeometryComponentEnum::NORMAL_Y), boundary)};
    state.materials = {Field(rows(SGNMaterialsComponentEnum::MASS), n),
        Field(rows(SGQMainMaterialsComponentEnum::MU), main), Field(0, boundary)};
}
}  // namespace

std::shared_ptr<CompiledState> compile_structure(Index n_count, PETData pets, SpaceData main, SpaceData boundary) {
    try {
        require_cuda_device();
        validate_structure(n_count, pets, main, boundary);
        auto state = std::make_shared<CompiledState>();
        state->n_count = n_count;
        state->pets = normalize_pets(pets);
        state->main = TenzorSpaceData(std::move(main));
        state->boundary = TenzorSpaceData(std::move(boundary));
        state->main_schedule = semgpu2::common::cu::build_q_block_schedule(state->main.q_node_offsets, THREADS_MAX);
        state->boundary_schedule = semgpu2::common::cu::build_q_block_schedule(state->boundary.q_node_offsets, THREADS_MAX);
        allocate_fields(*state);
        return state;
    } catch (const CudaError& error) {
        throw CompileError(SGCompileErrorCodeEnum::DEVICE_FAILURE, error.what());
    } catch (const std::exception& error) {
        throw CompileError(SGCompileErrorCodeEnum::STRUCTURE_FAILURE, error.what());
    }
}

void validate_structure(Index n_count, const PETData& pets, const SpaceData& main, const SpaceData& boundary) {
    if (n_count < 0) throw std::invalid_argument("n_count must be nonnegative");
    validate_pets(pets);
    validate_space(main, pets, n_count, 2);
    validate_space(boundary, pets, n_count, 1);
    // Check every canonical field extent before any host allocation.  This
    // permits structural int64 validation for large N without materializing
    // its numerical storage.
    checked_product(6, n_count);
    checked_product(3, n_count);
    checked_product(5, main.q_count());
    checked_product(3, boundary.q_count());
    checked_product(1, n_count);
    checked_product(3, main.q_count());
}
}  // namespace semgpu2_seismic_cu
