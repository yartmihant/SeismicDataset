#include "data.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>

namespace py = pybind11;
namespace SEMGPU2_SEISMIC_NAMESPACE {
void bind_constants(py::module_& module);
void bind_runtime(py::module_& module);

namespace {
template<class T>
py::array checked_array(py::handle object, int ndim, const char* label) {
    if (!py::isinstance<py::array>(object)) throw std::invalid_argument(std::string(label) + " must be a numpy array");
    auto array = py::reinterpret_borrow<py::array>(object);
    if (!array.dtype().is(py::dtype::of<T>()) || array.ndim() != ndim
        || !(array.flags() & py::array::c_style)) {
        throw std::invalid_argument(std::string(label) + " has an invalid dtype, rank or layout");
    }
    return array;
}

template<class T>
std::vector<T> copy_column(py::handle owner, const char* name) {
    const auto array = checked_array<T>(owner.attr(name), 1, name);
    if (!array.size()) return {};
    const T* begin = static_cast<const T*>(array.data());
    return std::vector<T>(begin, begin + array.size());
}

void validate_state_range(Index rows, Index columns, Index row_begin, Index row_count, Index row_step,
    Index column_begin, Index column_count, Index column_step) {
    const auto validate_axis = [](Index size, Index begin, Index count, Index step) {
        if (size < 0 || count < 0 || step == 0) {
            throw std::invalid_argument("State array range has an invalid shape, count or step");
        }
        if (count == 0) return;
        if (begin < 0 || begin >= size) throw std::out_of_range("State array slice is outside its shape");
        if (count == 1) return;
        const Index span = count - 1;
        if (step > 0) {
            if (step > (size - 1 - begin) / span) throw std::out_of_range("State array slice is outside its shape");
        } else if (step < -begin / span) {
            throw std::out_of_range("State array slice is outside its shape");
        }
    };
    validate_axis(rows, row_begin, row_count, row_step);
    validate_axis(columns, column_begin, column_count, column_step);
}

SpaceData read_space(py::handle owner) {
    SpaceData space;
    space.dim = owner.attr("dim").cast<Code>();
    space.degree = owner.attr("degree").cast<Code>();
    space.rank = owner.attr("rank").cast<Code>();
    space.pet_ids = copy_column<Code>(owner, "pet_ids");
    space.pet_offsets = copy_column<Index>(owner, "pet_offsets");
    space.p_nodes = copy_column<Index>(owner, "p_nodes");
    space.p_node_offsets = copy_column<Index>(owner, "p_node_offsets");
    space.q_node_offsets = copy_column<Index>(owner, "q_node_offsets");
    return space;
}

SpaceData read_main_space(py::handle source) {
    const auto mapping = py::reinterpret_borrow<py::object>(source);
    if (py::len(mapping) != 1) throw std::invalid_argument("Exactly one MAIN PQ space is required");
    for (py::handle key : mapping) {
        if (py::cast<Code>(key) == static_cast<Code>(SGSpaceEnum::MAIN)) return read_space(mapping[key]);
    }
    throw std::invalid_argument("MAIN space is required");
}

class StateArrayAdapter {
    std::shared_ptr<CompiledState> state_;
    std::array<Field, 2> CompiledState::* fields_;
    Field CompiledState::* standalone_ = nullptr;
    Index field_id_;

    const Field& field() const {
        if (standalone_) return state_.get()->*standalone_;
        return (state_.get()->*fields_).at(static_cast<std::size_t>(field_id_));
    }
    Field& field() {
        if (standalone_) return state_.get()->*standalone_;
        return (state_.get()->*fields_).at(static_cast<std::size_t>(field_id_));
    }
    void validate(Index row_begin, Index row_count, Index row_step,
        Index column_begin, Index column_count, Index column_step) const {
        const auto& source = field();
        validate_state_range(source.rows, source.columns, row_begin, row_count, row_step,
            column_begin, column_count, column_step);
    }
public:
    StateArrayAdapter(std::shared_ptr<CompiledState> state, std::array<Field, 2> CompiledState::* fields, Index field_id)
        : state_(std::move(state)), fields_(fields), field_id_(field_id) {}
    StateArrayAdapter(std::shared_ptr<CompiledState> state, Field CompiledState::* field)
        : state_(std::move(state)), fields_(nullptr), standalone_(field), field_id_(0) {}
    py::tuple shape() const { const auto& source = field(); return py::make_tuple(source.rows, source.columns); }
    py::array read(Index row_begin, Index row_count, Index row_step,
        Index column_begin, Index column_count, Index column_step) const {
        validate(row_begin, row_count, row_step, column_begin, column_count, column_step);
        py::array_t<Real> result({static_cast<py::ssize_t>(row_count), static_cast<py::ssize_t>(column_count)});
        const auto& source = field();
        for (Index row = 0; row < row_count; ++row) {
            for (Index column = 0; column < column_count; ++column) {
                *result.mutable_data(row, column) = source.values[(row_begin + row * row_step) * source.columns
                    + column_begin + column * column_step];
            }
        }
        return result;
    }
    void write(Index row_begin, Index row_count, Index row_step,
        Index column_begin, Index column_count, Index column_step, py::handle values) {
        const auto input = checked_array<Real>(values, 2, "State array values");
        validate(row_begin, row_count, row_step, column_begin, column_count, column_step);
        if (input.shape(0) != row_count || input.shape(1) != column_count) {
            throw std::invalid_argument("State array value shape differs from the selected slice");
        }
        auto& target = field();
        const auto* source = static_cast<const Real*>(input.data());
        for (Index row = 0; row < row_count; ++row) {
            for (Index column = 0; column < column_count; ++column) {
                target.values[(row_begin + row * row_step) * target.columns + column_begin + column * column_step]
                    = source[row * column_count + column];
            }
        }
    }
};

py::dict state_arrays(const std::shared_ptr<CompiledState>& state, std::array<Field, 2> CompiledState::* fields) {
    py::dict result;
    for (std::size_t id = 0; id < (state.get()->*fields).size(); ++id) {
        result[py::int_(id)] = StateArrayAdapter(state, fields, static_cast<Index>(id));
    }
    return result;
}

void set_stage_exception(const char* name, Code code, const char* message) {
    const auto type = py::module_::import(SEMGPU2_SEISMIC_CONSTANTS_MODULE).attr(name);
    const auto error = type(code, message);
    PyErr_SetObject(type.ptr(), error.ptr());
}
}  // namespace
}  // namespace SEMGPU2_SEISMIC_NAMESPACE

PYBIND11_MODULE(_native, module) {
    using namespace SEMGPU2_SEISMIC_NAMESPACE;
    module.doc() = "SEMGPU2 native stages: compiled constants, fixed-order ALine and geometry";
    bind_constants(module);
    py::register_local_exception_translator([](std::exception_ptr failure) {
        try {
            if (failure) std::rethrow_exception(failure);
        } catch (const CompileError& error) {
            set_stage_exception("SGCompileError", static_cast<Code>(error.code), error.what());
        } catch (const MeshError& error) {
            set_stage_exception("SGMeshError", static_cast<Code>(error.code), error.what());
        }
    });
    py::class_<CompiledState, std::shared_ptr<CompiledState>>(module, "NativeCompiledState")
        .def_readonly("n_count", &CompiledState::n_count)
        .def_readonly("meshed", &CompiledState::meshed)
        .def_property_readonly("solutions", [](const std::shared_ptr<CompiledState>& state) { return state_arrays(state, &CompiledState::solution); })
        .def_property_readonly("geometries", [](const std::shared_ptr<CompiledState>& state) { return state_arrays(state, &CompiledState::geometry); })
        .def_property_readonly("materials", [](const std::shared_ptr<CompiledState>& state) { return state_arrays(state, &CompiledState::materials); })
        .def_property_readonly("pml_history", [](const std::shared_ptr<CompiledState>& state) {
            return StateArrayAdapter(state, &CompiledState::pml_history);
        })
        .def_property_readonly("pml_entity_ids", [](const std::shared_ptr<CompiledState>& state) {
            py::array_t<Index> result(static_cast<py::ssize_t>(state->pml_entity_ids.size()));
            std::copy(state->pml_entity_ids.begin(), state->pml_entity_ids.end(), result.mutable_data());
            result.attr("setflags")(false);
            return result;
        });
    py::class_<StateArrayAdapter>(module, "StateArrayAdapter")
        .def_property_readonly("shape", &StateArrayAdapter::shape)
        .def("read", &StateArrayAdapter::read, py::arg("row_begin"), py::arg("row_count"), py::arg("row_step"), py::arg("column_begin"), py::arg("column_count"), py::arg("column_step"))
        .def("write", &StateArrayAdapter::write, py::arg("row_begin"), py::arg("row_count"), py::arg("row_step"), py::arg("column_begin"), py::arg("column_count"), py::arg("column_step"), py::arg("values"));
    module.def("mesh_geometry", [](CompiledState& state, py::object coordinates) {
        std::unique_lock<std::mutex> lock(state.operation_mutex, std::try_to_lock);
        if (!lock.owns_lock()) throw std::runtime_error("A solver stage is already active");
        if (state.run_active) throw std::runtime_error("A run is active");
        ++state.generation;
        state.meshed = false;
        try {
            const auto array = checked_array<Real>(coordinates, 2, "n_coords");
            if (array.shape(0) != 2 || array.shape(1) != state.n_count) throw std::invalid_argument("n_coords must have shape (2,N)");
            const Real* begin = static_cast<const Real*>(array.data());
            std::vector<Real> values(begin, begin + array.size());
            py::gil_scoped_release release;
            mesh_geometry(state, values.data(), static_cast<Index>(values.size()));
        } catch (const MeshError&) {
            throw;
        } catch (const std::exception& error) {
            throw MeshError(SGMeshErrorCodeEnum::GEOMETRY_FAILURE, error.what());
        }
    }, py::arg("state").none(false), py::arg("n_coords"));
    module.def("compile_structure", [](Index n_count, py::object spaces) {
        try {
            auto main = read_main_space(spaces);
            py::gil_scoped_release release;
            return compile_structure(n_count, std::move(main));
        } catch (const CompileError&) {
            throw;
        } catch (const std::exception& error) {
            throw CompileError(SGCompileErrorCodeEnum::STRUCTURE_FAILURE, error.what());
        }
    }, py::arg("n_count"), py::arg("pq_spaces"));
    module.def("validate_structure", [](Index n_count, py::object spaces) {
        try {
            const auto main = read_main_space(spaces);
            py::gil_scoped_release release;
            validate_structure(n_count, main);
        } catch (const CompileError&) {
            throw;
        } catch (const std::exception& error) {
            throw CompileError(SGCompileErrorCodeEnum::STRUCTURE_FAILURE, error.what());
        }
    }, py::arg("n_count"), py::arg("pq_spaces"));
    module.def("_validate_state_range", &validate_state_range,
        py::arg("rows"), py::arg("columns"), py::arg("row_begin"), py::arg("row_count"), py::arg("row_step"),
        py::arg("column_begin"), py::arg("column_count"), py::arg("column_step"));
    bind_runtime(module);
}
