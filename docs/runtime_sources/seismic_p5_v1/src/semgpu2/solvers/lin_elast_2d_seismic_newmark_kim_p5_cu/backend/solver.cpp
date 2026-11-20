#include "data.hpp"
#include "device_data.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace semgpu2_seismic_p5_cu {
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

void validate_space(const SpaceData& space, Index n_count, int rank) {
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
    const int width = rank == 2 ? PET_N * PET_N : PET_N;
    for (std::size_t element = 0; element < count; ++element) {
        const Index first = space.pet_offsets[element], last = space.pet_offsets[element + 1];
        if (last - first != rank) throw std::invalid_argument("Unsupported PET factor count");
        for (Index k = first; k < last; ++k) {
            if (space.pet_ids[k] != 0) throw std::invalid_argument("PET ID outside catalog");
        }
        if (width != space.p_node_offsets[element + 1] - space.p_node_offsets[element]
            || space.q_node_offsets[element + 1] != space.p_node_offsets[element + 1]) {
            throw std::invalid_argument("Element width must match the built-in p5 PET");
        }
    }
    for (Index node : space.p_nodes) {
        if (node < 0 || node >= n_count) throw std::invalid_argument("P-node mapping outside N");
    }
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

std::shared_ptr<CompiledState> compile_structure(Index n_count, SpaceData main, SpaceData boundary) {
    try {
        require_cuda_device();
        validate_structure(n_count, main, boundary);
        auto state = std::make_shared<CompiledState>();
        state->n_count = n_count;
        state->main = TenzorSpaceData(std::move(main), PET_N * PET_N);
        state->boundary = TenzorSpaceData(std::move(boundary), PET_N);
        allocate_fields(*state);
        return state;
    } catch (const CudaError& error) {
        throw CompileError(SGCompileErrorCodeEnum::DEVICE_FAILURE, error.what());
    } catch (const std::exception& error) {
        throw CompileError(SGCompileErrorCodeEnum::STRUCTURE_FAILURE, error.what());
    }
}

void validate_structure(Index n_count, const SpaceData& main, const SpaceData& boundary) {
    if (n_count < 0) throw std::invalid_argument("n_count must be nonnegative");
    validate_space(main, n_count, 2);
    validate_space(boundary, n_count, 1);
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
}  // namespace semgpu2_seismic_p5_cu
