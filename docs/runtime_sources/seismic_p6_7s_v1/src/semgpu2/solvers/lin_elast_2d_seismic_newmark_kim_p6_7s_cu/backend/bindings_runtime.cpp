#include "runtime.hpp"
#include "device_resources.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <thread>

namespace py = pybind11;
namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
template<class T> py::array array_of(py::handle source, int ndim, const char* name) {
    if (!py::isinstance<py::array>(source)) throw std::invalid_argument(std::string(name) + " must be a numpy array");
    auto array = py::reinterpret_borrow<py::array>(source);
    if (!array.dtype().is(py::dtype::of<T>()) || array.ndim() != ndim || !(array.flags() & py::array::c_style))
        throw std::invalid_argument(std::string(name) + " has invalid dtype, rank or layout");
    return array;
}
template<class T> std::vector<T> column(py::handle source, const char* name) {
    const auto array = array_of<T>(source.attr(name), 1, name);
    if (!array.size()) return {};
    const auto* pointer = static_cast<const T*>(array.data());
    return {pointer, pointer + array.size()};
}

std::vector<PhysicsData> read_physics(py::handle mapping) {
    std::vector<PhysicsData> result;
    const auto object = py::reinterpret_borrow<py::object>(mapping);
    for (py::handle key : object) {
        auto source = object[key];
        if (!source.attr("active").cast<bool>()) continue;
        PhysicsData field;
        field.type_code = source.attr("type_code").cast<Code>();
        if (field.type_code != static_cast<Code>(SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN)
            && field.type_code != static_cast<Code>(SGPhysicsTypeEnum::PML))
            throw std::invalid_argument("p6_7s accepts only MAIN material (11000) and PML (31100); the source is built in");
        if (!source.attr("t_begin").is_none() || !source.attr("t_end").is_none() || !source.attr("interp_code").is_none())
            throw std::invalid_argument("p6_7s material and PML must be static");
        field.space_id = source.attr("space_id").cast<Code>();
        field.entity_ids = column<Index>(source, "entity_ids");
        const auto values = array_of<Real>(source.attr("values"), 2, "physics values");
        field.component_count = values.shape(0);
        if (values.size()) {
            const auto* pointer = static_cast<const Real*>(values.data());
            field.values.assign(pointer, pointer + values.size());
        }
        result.push_back(std::move(field));
    }
    return result;
}
SolverConfig read_config(py::handle object) {
    SolverConfig result;
    const auto optional_index = [&](const char* name, std::optional<Index>& target) { const auto value = object.attr(name); if (!value.is_none()) target = value.cast<Index>(); };
    const auto optional_real = [&](const char* name, std::optional<Real>& target) { const auto value = object.attr(name); if (!value.is_none()) target = value.cast<Real>(); };
    optional_index("tacts_count", result.tacts_count);
    optional_index("steps_count", result.steps_count);
    optional_index("steps_per_tact", result.steps_per_tact);
    optional_index("max_steps", result.max_steps);
    result.t_begin = object.attr("t_begin").cast<Real>();
    result.t_end = object.attr("t_end").cast<Real>();
    optional_real("step_dt", result.step_dt);
    optional_real("tact_dt", result.tact_dt);
    optional_real("cfl_safety", result.cfl_safety);
    result.check_finite_each_step = object.attr("check_finite_each_step").cast<bool>();
    result.kim_beta = object.attr("kim_beta").cast<Real>();
    result.kim_gamma = object.attr("kim_gamma").cast<Real>();
    return result;
}
TimeGridBuilder make_time_grid_builder(py::object config_snapshot) {
    return [snapshot = std::move(config_snapshot)](Real step_bound) {
        py::gil_scoped_acquire gil;
        try {
            const py::object grid = py::module_::import("semgpu2.solvers.common.time_grid")
                .attr("build_time_grid")(snapshot, step_bound);
            return TimeGrid{
                grid.attr("step_dt").cast<Real>(),
                grid.attr("tact_dt").cast<Real>(),
                grid.attr("tacts_count").cast<Index>(),
                grid.attr("steps_count").cast<Index>(),
                grid.attr("steps_per_tact").cast<Index>(),
            };
        } catch (py::error_already_set& error) {
            throw std::invalid_argument(error.what());
        }
    };
}
py::object spectrum_dto(const SpectrumEstimate& values, const TimeGrid& grid) {
    py::array_t<Real> by_element(static_cast<py::ssize_t>(values.by_element.size()));
    std::copy(values.by_element.begin(), values.by_element.end(), by_element.mutable_data());
    by_element.attr("setflags")(false);
    const auto type = py::module_::import(SEMGPU2_SEISMIC_CONSTANTS_MODULE).attr("SGSpectrumData");
    return type(by_element, values.stiffness, grid.step_dt, grid.tact_dt, grid.tacts_count, grid.steps_count, grid.steps_per_tact, values.damping);
}

struct ReceiverPackages { std::vector<Receiver> values; };

ReceiverPackages read_receivers(const CompiledState& state, py::object sequence) {
    ReceiverPackages result;
    for (py::handle source : sequence) {
        Receiver package;
        package.space_id = source.attr("space_id").cast<Code>();
        package.begin_tact = source.attr("begin_tact").cast<Index>();
        package.end_tact = source.attr("end_tact").cast<Index>();
        package.stride = source.attr("stride").cast<Index>();
        package.comp_codes = column<Code>(source, "comp_codes");
        package.sens_entity_ids = column<Index>(source, "sens_entity_ids");
        package.sens_frame_idxs = column<Index>(source, "sens_frame_idxs");
        package.sens_weights = column<Real>(source, "sens_weights");
        package.frame_size = source.attr("frame_size").cast<Index>();
        package.frame_buf_size = source.attr("frame_buf_size").cast<Index>();
        result.values.push_back(std::move(package));
    }
    result.values = prepare_receivers(std::move(result.values), state);
    return result;
}

struct NativeHandle {
    std::weak_ptr<RunAuthority> authority;
    std::shared_ptr<Completion> completion;
    void command(RunCommand kind, Code code) const {
        const auto active = authority.lock();
        if (!active) throw std::runtime_error("Run handle is no longer active");
        active->enqueue(kind, code);
    }
    void handler_failure() const { if (const auto active = authority.lock()) active->handler_failure(); }
    py::tuple wait() const {
        try {
            RunOutcome result;
            { py::gil_scoped_release release; result = completion->wait(); }
            return py::make_tuple(result.code, result.tact, result.current_time, result.steps_count, result.aborted, false, "");
        } catch (const RunFailure& failure) {
            return py::make_tuple(failure.code, failure.tact, failure.current_time, failure.steps_count, true, true, failure.what());
        }
    }
};

class NativeSession : public std::enable_shared_from_this<NativeSession> {
    std::shared_ptr<PreparedState> prepared_;
    py::function callback_, finished_;
    std::vector<Receiver> receivers_;
    std::shared_ptr<RunAuthority> authority_ = std::make_shared<RunAuthority>();
    bool started_ = false;

    void clear_anchors() {
        callback_ = py::function(); finished_ = py::function();
    }
    void execute(bool background) {
        RunOutcome outcome;
        std::optional<RunFailure> failure;
        try {
            outcome = run(*prepared_, receivers_, *authority_, !background,
                [&](Index tact, Real time, Index steps_count, Code event, Index index, Real* frame, Index length) {
                    py::gil_scoped_acquire gil;
                    py::array_t<Real> view;
                    if (length) {
                        py::capsule borrowed(frame, [](void*) {});
                        view = py::array_t<Real>({static_cast<py::ssize_t>(length)},
                            {static_cast<py::ssize_t>(sizeof(Real))}, frame, borrowed);
                    } else view = py::array_t<Real>(0);
                    try { callback_(tact, time, steps_count, event, index, view); }
                    catch (py::error_already_set& error) {
                        authority_->handler_failure();
                        error.discard_as_unraisable("native receiver callback");
                    }
                });
        } catch (const RunFailure& error) {
            failure = error;
        } catch (const std::exception& error) {
            failure.emplace(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), 0, prepared_->config.t_begin, error.what());
        } catch (...) {
            failure.emplace(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), 0, prepared_->config.t_begin, "Unknown native run failure");
        }
        authority_->close_commands();
        prepared_->structure->run_active = false;
        {
            py::gil_scoped_acquire gil;
            try { finished_(failure ? true : outcome.aborted, failure.has_value()); }
            catch (py::error_already_set& error) { error.discard_as_unraisable("native completion callback"); }
            clear_anchors();
        }
        prepared_.reset();
        receivers_.clear();
        if (failure) authority_->completion()->publish_error(*failure);
        else authority_->completion()->publish(outcome);
    }
public:
    NativeSession(std::shared_ptr<PreparedState> prepared,
        py::object receivers, py::function callback, py::function finished)
        : prepared_(std::move(prepared)),
          callback_(std::move(callback)), finished_(std::move(finished)) {
        if (!prepared_ || !prepared_->structure || !prepared_->physics)
            throw std::invalid_argument("Session requires a prepared model");
        if (!receivers.is_none()) receivers_ = receivers.cast<ReceiverPackages>().values;
    }
    ~NativeSession() { py::gil_scoped_acquire gil; clear_anchors(); }
    NativeHandle handle() const { return {authority_, authority_->completion()}; }
    void start(bool sync) {
        if (started_) throw std::runtime_error("Native run already started");
        auto& state = *prepared_->structure;
        {
            std::unique_lock<std::mutex> lock(state.operation_mutex, std::try_to_lock);
            if (!lock.owns_lock() || state.run_active) throw std::runtime_error("A solver operation is active");
            if (!state.meshed || prepared_->generation != state.generation) throw std::runtime_error("Prepared state has been invalidated");
            state.run_active = true;
        }
        started_ = true;
        if (sync) {
            py::gil_scoped_release release;
            execute(false);
        } else {
            auto self = shared_from_this();
            try { std::thread([self] { self->execute(true); }).detach(); }
            catch (...) { state.run_active = false; started_ = false; throw; }
        }
    }
};
}  // namespace

py::object empty_spectrum(const std::shared_ptr<CompiledState>& state) {
    SpectrumEstimate values;
    values.by_element.resize(array_size<Real>(state->main.element_count()), 0.0);
    return spectrum_dto(values, TimeGrid{});
}

void bind_runtime(py::module_& module) {
    module.def("empty_spectrum", &empty_spectrum, py::arg("state").none(false));
    py::register_local_exception_translator([](std::exception_ptr failure) {
        try { if (failure) std::rethrow_exception(failure); }
        catch (const PrepareError& error) {
            const auto type = py::module_::import(SEMGPU2_SEISMIC_CONSTANTS_MODULE).attr("SGPrepareError");
            const auto value = type(static_cast<Code>(error.code), error.what());
            PyErr_SetObject(type.ptr(), value.ptr());
        }
    });
    py::class_<PreparedState, std::shared_ptr<PreparedState>>(module, "PreparedState")
        .def_property_readonly("spectrum", [](const PreparedState& state) { return spectrum_dto(state.spectrum, state.grid); });
    py::class_<ReceiverPackages>(module, "ReceiverPackages");
    module.def("prepare_model", [](std::shared_ptr<CompiledState> state, py::object physics, py::object config) {
        std::unique_lock<std::mutex> lock(state->operation_mutex, std::try_to_lock);
        if (!lock.owns_lock() || state->run_active) throw std::runtime_error("A solver operation is active");
        ++state->generation;
        SolverConfig native_config;
        try { native_config = read_config(config); }
        catch (const std::exception& error) { throw PrepareError(SGPrepareErrorCodeEnum::SPECTRUM_FAILURE, error.what()); }
        py::object config_snapshot;
        try { config_snapshot = py::module_::import("dataclasses").attr("replace")(config); }
        catch (const std::exception& error) { throw PrepareError(SGPrepareErrorCodeEnum::SPECTRUM_FAILURE, error.what()); }
        TimeGridBuilder time_grid = make_time_grid_builder(std::move(config_snapshot));
        std::vector<PhysicsData> fields;
        try { fields = read_physics(physics); }
        catch (const std::exception& error) { throw PrepareError(SGPrepareErrorCodeEnum::PHYSICS_FAILURE, error.what()); }
        std::shared_ptr<PreparedState> prepared;
        {
            py::gil_scoped_release release;
            prepared = prepare_model(std::move(state), std::move(fields), native_config, time_grid);
        }
        return prepared;
    }, py::arg("state"), py::arg("physics"), py::arg("config"));
    module.def("prepare_receivers", [](const std::shared_ptr<CompiledState>& state, py::object receivers) {
        return read_receivers(*state, receivers);
    }, py::arg("state"), py::arg("receivers"));
    py::class_<NativeHandle>(module, "Handle")
        .def("save", [](const NativeHandle& self) { self.command(RunCommand::SAVE, 0); })
        .def("complete", [](const NativeHandle& self, Code code) { self.command(RunCommand::COMPLETE, code); })
        .def("abort", [](const NativeHandle& self, Code code) { self.command(RunCommand::ABORT, code); })
        .def("_handler_failure", &NativeHandle::handler_failure)
        .def("wait", &NativeHandle::wait);
    py::class_<NativeSession, std::shared_ptr<NativeSession>>(module, "Session")
        .def(py::init<std::shared_ptr<PreparedState>, py::object, py::function, py::function>())
        .def("handle", &NativeSession::handle)
        .def("start", &NativeSession::start);
}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
