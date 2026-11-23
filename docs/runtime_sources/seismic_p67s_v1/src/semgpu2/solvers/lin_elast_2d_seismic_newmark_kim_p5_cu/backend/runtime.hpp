#pragma once

#include "data.hpp"
#include "physics.hpp"
#include "receivers.hpp"
#include "spectrum.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <optional>

namespace SEMGPU2_SEISMIC_NAMESPACE {

// Mirrors the shared SolverAPI command/event protocol at the native boundary.
enum class RunCommand : Code { SAVE = 0, COMPLETE = 1, ABORT = 2 };
enum class ReceiverEvent : Code { INITIAL = 0, PROCESS = 1, POSITIVE = 2, NEGATIVE = 3 };

struct SolverConfig {
    std::optional<Index> tacts_count, steps_count, steps_per_tact, max_steps;
    Real t_begin = DEFAULT_T_BEGIN, t_end = DEFAULT_T_END;
    std::optional<Real> step_dt, tact_dt, cfl_safety;
    bool check_finite_each_step = false;
    Real kim_beta = DEFAULT_KIM_BETA, kim_gamma = DEFAULT_KIM_GAMMA;
};

struct TimeGrid {
    Real step_dt = 0.0, tact_dt = 0.0;
    Index tacts_count = 0, steps_count = 0, steps_per_tact = 0;
};

struct PreparedState {
    std::shared_ptr<CompiledState> structure;
    std::shared_ptr<PreparedPhysics> physics;
    SolverConfig config;
    SpectrumEstimate spectrum;
    TimeGrid grid;
    Index generation = 0;
    Real source_position(Index completed_steps) const {
        const Index next = completed_steps + 1;
        return next == grid.steps_count ? config.t_end : config.t_begin + static_cast<Real>(next) * grid.step_dt;
    }
    const std::map<Index, PreparedField>& run_sources() const { return physics->primary; }
};

struct PrepareError : std::runtime_error {
    SGPrepareErrorCodeEnum code;
    PrepareError(SGPrepareErrorCodeEnum value, const std::string& message)
        : std::runtime_error(message), code(value) {}
};

// The shared Python layer owns the exact rational time-grid construction.  CUDA
// only supplies the CFL step bound after its spectral estimate is available.
using TimeGridBuilder = std::function<TimeGrid(Real step_bound)>;

std::shared_ptr<PreparedState> prepare_model(std::shared_ptr<CompiledState> state,
    const std::vector<PhysicsData>& physics, const SolverConfig& config,
    const TimeGridBuilder& time_grid);

struct RunOutcome {
    Code code = static_cast<Code>(SGRunCodeEnum::TIME_END);
    Index tact = 0;
    Real current_time = 0.0;
    Index steps_count = 0;
    bool aborted = false;
};

struct RunFailure : std::runtime_error {
    Code code;
    Index tact;
    Real current_time;
    Index steps_count;
    RunFailure(Code value, Index tact_in, Real time, const std::string& message, Index steps_in = 0)
        : std::runtime_error(message), code(value), tact(tact_in), current_time(time), steps_count(steps_in) {}
};

struct Command { RunCommand kind; Code code; };

// The completion owns only a frozen outcome, not a solver or its resources.
class Completion {
    mutable std::mutex mutex_;
    std::condition_variable ready_condition_;
    bool ready_ = false, failed_ = false;
    RunOutcome outcome_;
    std::string message_;
public:
    void publish(const RunOutcome& outcome);
    void publish_error(const RunFailure& failure);
    RunOutcome wait();
};

class RunAuthority {
    mutable std::mutex mutex_;
    std::deque<Command> commands_;
    bool closed_ = false;
    std::string device_failure_;
    std::shared_ptr<Completion> completion_ = std::make_shared<Completion>();
public:
    void enqueue(RunCommand kind, Code code);
    std::vector<Command> drain_commands();
    void close_commands();
    void handler_failure();
    void device_failure(const std::string& message);
    std::string device_failure() const;
    const std::shared_ptr<Completion>& completion() const { return completion_; }
};

using ReceiverCallback = std::function<void(Index tact, Real time, Index steps_count, Code event,
    Index receiver, Real* frame, Index length)>;

RunOutcome run(const PreparedState& prepared,
    const std::vector<Receiver>& receivers, RunAuthority& authority,
    bool force_sync, const ReceiverCallback& callback);

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
