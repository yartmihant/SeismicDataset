#include <runtime.hpp>
#include <stepping.hpp>

#include <thread>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
class Event {
    CleanupStatus& cleanup_;
    cudaEvent_t event_ = nullptr;
public:
    explicit Event(CleanupStatus& cleanup) : cleanup_(cleanup) {
        cuda_check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "cudaEventCreate");
    }
    ~Event() { if (event_) cleanup_.record(cudaEventDestroy(event_)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void record(cudaStream_t stream) const { cuda_check(cudaEventRecord(event_, stream), "cudaEventRecord"); }
    void wait(cudaStream_t stream) const { cuda_check(cudaStreamWaitEvent(stream, event_, 0), "cudaStreamWaitEvent"); }
    void synchronize() const { cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize"); }
};

class PinnedBuffer {
    CleanupStatus& cleanup_;
    Real* pointer_ = nullptr;
public:
    PinnedBuffer(CleanupStatus& cleanup, Index length) : cleanup_(cleanup) {
        const auto count = array_size<Real>(length);
        if (count) cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pointer_), count * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc");
    }
    ~PinnedBuffer() { if (pointer_) cleanup_.record(cudaFreeHost(pointer_)); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    Real* get() const { return pointer_; }
};

struct FrameSlot {
    // Host target outlives stream and therefore every outstanding D2H.
    PinnedBuffer host;
    CudaStream stream;
    DeviceBuffer<Real> device;
    Event read_complete, d2h_complete;
    Index tact = 0;
    Real time = 0.0;
    Index steps_count = 0;
    Code event = 0;
    bool busy = false;
    FrameSlot(CleanupStatus& cleanup, Index length)
        : host(cleanup, length), stream(cleanup), device(cleanup, length),
          read_complete(cleanup), d2h_complete(cleanup) {}
};

class ReceiverChannel {
    const Receiver& package_;
    const ReceiverDevicePlan& plan_;
    Index receiver_;
    Index length_;
    bool asynchronous_;
    RunAuthority& authority_;
    const ReceiverCallback& callback_;
    std::vector<std::unique_ptr<FrameSlot>> slots_;
    std::mutex mutex_;
    std::condition_variable jobs_ready_, free_ready_;
    std::deque<std::size_t> jobs_;
    bool stopping_ = false;
    std::thread worker_;

    std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        free_ready_.wait(lock, [&] {
            for (const auto& slot : slots_) if (!slot->busy) return true;
            return false;
        });
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            if (!slots_[index]->busy) { slots_[index]->busy = true; return index; }
        }
        throw std::logic_error("Receiver ring has no free slot");
    }
    void release(std::size_t index) {
        { std::lock_guard<std::mutex> lock(mutex_); slots_[index]->busy = false; }
        free_ready_.notify_all();
    }
    void deliver(FrameSlot& slot) {
        try {
            slot.d2h_complete.synchronize();
            callback_(slot.tact, slot.time, slot.steps_count, slot.event, receiver_, slot.host.get(), length_);
        } catch (const CudaError& error) {
            authority_.device_failure(error.what());
        } catch (...) {
            authority_.handler_failure();
        }
    }
    void worker_loop() {
        for (;;) {
            std::size_t index;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                jobs_ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty() && stopping_) return;
                index = jobs_.front(); jobs_.pop_front();
            }
            deliver(*slots_[index]);
            release(index);
        }
    }
public:
    ReceiverChannel(CleanupStatus& cleanup, const Receiver& package, const ReceiverDevicePlan& plan,
        Index receiver, bool force_sync, RunAuthority& authority, const ReceiverCallback& callback)
        : package_(package), plan_(plan), receiver_(receiver),
          length_(checked_product(package.frame_size, static_cast<Index>(package.comp_codes.size()))),
          asynchronous_(!force_sync && package.frame_buf_size > 0), authority_(authority), callback_(callback) {
        const Index capacity = package.frame_buf_size;
        if (asynchronous_ && capacity == std::numeric_limits<Index>::max()) throw std::overflow_error("Receiver ring capacity overflow");
        const Index slots = asynchronous_ ? capacity + 1 : 1;
        slots_.reserve(array_size<std::unique_ptr<FrameSlot>>(slots));
        for (Index index = 0; index < slots; ++index) slots_.push_back(std::make_unique<FrameSlot>(cleanup, length_));
        if (asynchronous_) worker_ = std::thread([this] { worker_loop(); });
    }
    ~ReceiverChannel() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        jobs_ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    bool scheduled(Index tact) const { return plan_.scheduled(tact); }
    bool terminal_receiver() const { return package_.end_tact == -1; }
    std::size_t capture(const Event& ready, DeviceExecutionContext& device, Index tact, Real time, Index steps_count, ReceiverEvent event) {
        const auto index = acquire();
        auto& slot = *slots_[index];
        slot.tact = tact; slot.time = time; slot.steps_count = steps_count; slot.event = static_cast<Code>(event);
        try {
            ready.wait(slot.stream.get());
            plan_.capture(device, slot.device.get(), slot.stream.get());
            slot.read_complete.record(slot.stream.get());
            // Only gather must precede the next numerical write; D2H and
            // the callback may overlap with the next complete model step.
            slot.read_complete.wait(device.stream());
            if (length_) cuda_check(cudaMemcpyAsync(slot.host.get(), slot.device.get(), array_size<Real>(length_) * sizeof(Real),
                cudaMemcpyDeviceToHost, slot.stream.get()), "Receiver D2H");
            slot.d2h_complete.record(slot.stream.get());
            return index;
        } catch (...) { release(index); throw; }
    }
    void dispatch(std::size_t index) {
        auto& slot = *slots_[index];
        if (asynchronous_) {
            { std::lock_guard<std::mutex> lock(mutex_); jobs_.push_back(index); }
            jobs_ready_.notify_one();
        } else { deliver(slot); release(index); }
    }
    void cancel(std::size_t index) { release(index); }
    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        free_ready_.wait(lock, [&] {
            for (const auto& slot : slots_) if (slot->busy) return false;
            return jobs_.empty();
        });
    }
};

RunOutcome run_body(CleanupStatus& cleanup, const PreparedState& prepared,
    const std::vector<Receiver>& packages, RunAuthority& authority, bool force_sync,
    const ReceiverCallback& callback, Index& last_complete, Real& last_time, Index& last_steps) {
    const auto& cfg = prepared.config;
    const auto& grid = prepared.grid;
    auto& physics = *prepared.physics;
    auto& state = *prepared.structure;
    DeviceExecutionContext device(cleanup, state, state.materials[0].values,
        state.materials[1].values, prepared.run_sources(), physics.pml, state.pml_history.values, cfg.check_finite_each_step);
    Event fields_ready(cleanup);
    std::vector<std::unique_ptr<ReceiverDevicePlan>> plans;
    std::vector<std::unique_ptr<ReceiverChannel>> channels;
    // All channels are destroyed/joined before device/plan resources.
    for (Index index = 0; index < static_cast<Index>(packages.size()); ++index) {
        const auto& package = packages[static_cast<std::size_t>(index)];
        plans.push_back(std::make_unique<ReceiverDevicePlan>(cleanup, package, *prepared.structure, device.stream()));
        channels.push_back(std::make_unique<ReceiverChannel>(cleanup, package, *plans.back(), index, force_sync, authority, callback));
    }
    Index completed_steps = 0;
    auto current_time = [&] {
        return completed_steps == grid.steps_count ? cfg.t_end
            : cfg.t_begin + static_cast<Real>(completed_steps) * grid.step_dt;
    };
    bool step_limited = false;
    auto drain = [&] { for (auto& channel : channels) channel->drain(); };
    auto commands = [&](Index tact) -> std::optional<RunOutcome> {
        for (const auto& command : authority.drain_commands()) {
            if (command.kind == RunCommand::SAVE) {
                device.copy_solution_to_host(prepared.structure->solution[0]);
                if (!state.pml_history.values.empty()) device.copy_pml_history_to_host(state.pml_history.values);
            } else {
                authority.close_commands();
                return RunOutcome{command.code, tact, last_time, last_steps, command.kind == RunCommand::ABORT};
            }
        }
        return {};
    };
    const bool underconfigured = grid.tacts_count <= 0 || grid.steps_per_tact <= 0;
    // Tact 0 publishes the supplied state and zero response, as in NB.
    for (Index tact = 0;; ++tact) {
        if (tact) {
            const Index remaining_steps = grid.steps_count - completed_steps;
            const Index tact_steps = std::min(grid.steps_per_tact, remaining_steps);
            for (Index step = 0; step < tact_steps; ++step) {
                const Index next_steps = completed_steps + 1;
                const bool is_step_limit = cfg.max_steps && next_steps >= *cfg.max_steps && next_steps < grid.steps_count;
#ifdef SEMGPU2_SEISMIC_FUSE_INTERNAL_STEPS
                // A tact boundary is observable by receivers and commands.  The
                // final step therefore keeps the corrected state in memory.
                const bool starts_predicted = !cfg.check_finite_each_step && step > 0;
                const bool predicts_next = !cfg.check_finite_each_step && !is_step_limit && step + 1 < tact_steps;
                if (!device.advance_internal_step(prepared.source_position(completed_steps), grid.step_dt, cfg.kim_beta,
                    cfg.kim_gamma, cfg.check_finite_each_step, starts_predicted, predicts_next))
#else
                if (!device.advance_internal_step(prepared.source_position(completed_steps), grid.step_dt, cfg.kim_beta,
                    cfg.kim_gamma, cfg.check_finite_each_step))
#endif
                    throw RunFailure(static_cast<Code>(SGRunCodeEnum::NUMERICAL_DIVERGENCE), last_complete, last_time, "Non-finite native step", last_steps);
                completed_steps = next_steps;
                if (is_step_limit) { step_limited = true; break; }
            }
        }
        const bool terminal = underconfigured || step_limited || completed_steps == grid.steps_count;
        const auto event = (underconfigured || step_limited) ? ReceiverEvent::NEGATIVE
            : (terminal ? ReceiverEvent::POSITIVE : (tact ? ReceiverEvent::PROCESS : ReceiverEvent::INITIAL));
        device.synchronize();
        fields_ready.record(device.stream());
        struct PendingFrame { ReceiverChannel* channel; std::size_t slot; };
        std::vector<PendingFrame> pending;
        pending.reserve(channels.size());
        try {
            for (auto& channel : channels) {
                if (channel->scheduled(tact) || (terminal && channel->terminal_receiver())) {
                    pending.push_back({channel.get(), channel->capture(fields_ready, device, tact, current_time(), completed_steps, event)});
                }
            }
        } catch (...) {
            for (const PendingFrame& frame : pending) frame.channel->cancel(frame.slot);
            throw;
        }
        std::size_t dispatched = 0;
        try {
            for (; dispatched < pending.size(); ++dispatched) {
                const PendingFrame& frame = pending[dispatched];
                frame.channel->dispatch(frame.slot);
            }
        } catch (...) {
            for (; dispatched < pending.size(); ++dispatched) {
                const PendingFrame& frame = pending[dispatched];
                frame.channel->cancel(frame.slot);
            }
            throw;
        }
        last_complete = tact;
        last_time = current_time();
        last_steps = completed_steps;
        if (terminal) drain();
        auto selected = commands(tact);
        if (selected) {
            drain();
            const auto device_error = authority.device_failure();
            if (!selected->aborted && !device_error.empty()) throw RunFailure(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), tact, last_time, device_error, last_steps);
            return *selected;
        }
        if (terminal) {
            // Accepted terminal frames have drained. Their FIFO commands
            // precede defaults and may have selected a different outcome.
            if (underconfigured) {
                authority.enqueue(RunCommand::ABORT, static_cast<Code>(SGRunCodeEnum::UNDERCONFIGURED));
            } else if (step_limited) {
                authority.enqueue(RunCommand::SAVE, 0);
                authority.enqueue(RunCommand::ABORT, static_cast<Code>(SGRunCodeEnum::STEP_LIMIT));
            } else {
                authority.enqueue(RunCommand::SAVE, 0);
                authority.enqueue(RunCommand::COMPLETE, static_cast<Code>(SGRunCodeEnum::TIME_END));
            }
            selected = commands(tact);
            if (!selected) throw std::logic_error("Missing native terminal outcome");
            return *selected;
        }
    }
}
}  // namespace

RunOutcome run(const PreparedState& prepared, const std::vector<Receiver>& receivers,
    RunAuthority& authority, bool force_sync, const ReceiverCallback& callback) {
    CleanupStatus cleanup;
    Index last_complete = 0;
    Real last_time = prepared.config.t_begin;
    Index last_steps = 0;
    std::optional<RunOutcome> outcome;
    std::exception_ptr failure;
    try { outcome = run_body(cleanup, prepared, receivers, authority, force_sync, callback, last_complete, last_time, last_steps); }
    catch (...) { failure = std::current_exception(); }
    authority.close_commands();
    const Real time = last_time;
    // Preserve the first selected failure. Cleanup cannot turn success into
    // a successful report, but cannot replace an already accepted ABORT.
    if (failure) {
        try { std::rethrow_exception(failure); }
        catch (const RunFailure&) { throw; }
        catch (const NumericalFailure& error) { throw RunFailure(static_cast<Code>(SGRunCodeEnum::NUMERICAL_DIVERGENCE), last_complete, time, error.what(), last_steps); }
        catch (const std::exception& error) { throw RunFailure(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), last_complete, time, error.what(), last_steps); }
    }
    if (!outcome) throw RunFailure(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), last_complete, time, "Missing native outcome", last_steps);
    if (!outcome->aborted && cleanup.error != cudaSuccess)
        throw RunFailure(static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE), last_complete, time, cudaGetErrorString(cleanup.error), last_steps);
    return *outcome;
}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
