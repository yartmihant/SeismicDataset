#pragma once

#include "constants.hpp"
#include "pet_data_tenzor.hpp"

#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace semgpu2_seismic_cu {
using semgpu2::common::cu::tenzor::PETView;

inline Index checked_product(Index a, Index b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<Index>::max() / b)) {
        throw std::overflow_error("Array shape exceeds int64 capacity");
    }
    return a * b;
}

template<class T>
std::size_t array_size(Index count) {
    if (count < 0 || static_cast<std::uint64_t>(count) > std::vector<T>().max_size()) {
        throw std::overflow_error("Array allocation size exceeds host capacity");
    }
    return static_cast<std::size_t>(count);
}

struct Field {
    Index rows = 0;
    Index columns = 0;
    std::vector<Real> values;
    Field() = default;
    Field(Index rows_in, Index columns_in)
        : rows(rows_in), columns(columns_in),
          values(array_size<Real>(checked_product(rows, columns)), 0.0) {}
};

struct PETData {
    std::vector<Index> p_node_counts, q_node_counts, orb_counts, symm_counts;
    std::vector<Code> dims;
    std::vector<Real> pq_values, pq_grads, q_weights, symm_grad_ops;
    std::vector<Index> pq_value_offsets, pq_grad_offsets, q_weight_offsets;
    std::vector<Code> q_node_orbs, q_node_rep_symms;
    std::vector<Index> q_node_orb_offsets, q_node_rep_symm_offsets;
    std::vector<Index> symm_grad_op_offsets, p_symm_inv_nodes, p_symm_inv_node_offsets;
};

// The structural input remains separate from the normalized execution artifact.
struct TenzorPETData {
    Index node_count = 0;
    std::vector<Real> pq_grads, q_weights;
};

struct SpaceData {
    Code dim = 2;
    Code degree = 1;
    Code rank = 1;
    std::vector<Code> pet_ids;
    std::vector<Index> pet_offsets{0}, p_nodes, p_node_offsets{0}, q_node_offsets{0};
    Index element_count() const { return static_cast<Index>(pet_offsets.size()) - 1; }
    Index q_count() const { return q_node_offsets.back(); }
};

struct TenzorSpaceData {
    Code rank = 1;
    int nodes_per_element = 1;
    std::vector<Index> p_nodes;

    Index q_count() const { return static_cast<Index>(p_nodes.size()); }
    Index element_count() const { return q_count() / nodes_per_element; }

    TenzorSpaceData() = default;
    TenzorSpaceData(SpaceData&& input, int width)
        : rank(input.rank), nodes_per_element(width), p_nodes(std::move(input.p_nodes)) {}
};

struct CompiledState {
    std::mutex operation_mutex;
    std::atomic<bool> run_active{false};
    Index generation = 0;
    Index n_count = 0;
    TenzorPETData pets;
    TenzorSpaceData main, boundary;
    std::array<Field, 3> solution, geometry, materials;
    bool meshed = false;
};

struct CompileError : std::runtime_error {
    SGCompileErrorCodeEnum code;
    CompileError(SGCompileErrorCodeEnum code_in, const std::string& message)
        : std::runtime_error(message), code(code_in) {}
};
struct MeshError : std::runtime_error {
    SGMeshErrorCodeEnum code;
    MeshError(SGMeshErrorCodeEnum code_in, const std::string& message)
        : std::runtime_error(message), code(code_in) {}
};

struct NumericalFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::shared_ptr<CompiledState> compile_structure(Index n_count, PETData pets, SpaceData main, SpaceData boundary);
void validate_structure(Index n_count, const PETData& pets, const SpaceData& main, const SpaceData& boundary);
void mesh_geometry(CompiledState& state, const Real* coordinates, Index coordinate_count);
}  // namespace semgpu2_seismic_cu
