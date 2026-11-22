#pragma once

#include "data.hpp"
#include "device_resources.hpp"

#include <map>
#include <memory>
#include <vector>

namespace semgpu2_seismic_cu {
class DeviceExecutionContext;

// Native counterpart of one SGReceiverData.  The entity list retains every
// occurrence: positive-space entries address elements and their offsets span
// complete P blocks; N entries have one value each.
struct Receiver {
    Code space_id = static_cast<Code>(SGSpaceEnum::N);
    Index begin_tact = -1, end_tact = -1, stride = 1;
    std::vector<Code> comp_codes;
    std::vector<Index> sens_entity_ids, sens_entity_offsets, sens_frame_idxs;
    std::vector<Real> sens_weights;
    Index frame_size = 0, frame_buf_size = 0;
};

struct ReceiverDevicePlan;

// Validation/setup snapshots one ordered receiver sequence for one run.
// All sparse writes are checked against their target.
std::vector<Receiver> prepare_receivers(
    const std::vector<Receiver>& receivers, const CompiledState& state);

// Device view/upload and numerical capture are isolated from FIFO mechanics.
// The caller owns output_frame and keeps it live until its D2H event completes.
struct ReceiverDevicePlan {
    explicit ReceiverDevicePlan(CleanupStatus& cleanup, const Receiver& host, const CompiledState& state, cudaStream_t stream);
    ~ReceiverDevicePlan();
    ReceiverDevicePlan(const ReceiverDevicePlan&) = delete;
    ReceiverDevicePlan& operator=(const ReceiverDevicePlan&) = delete;
    ReceiverDevicePlan(ReceiverDevicePlan&&) = delete;
    ReceiverDevicePlan& operator=(ReceiverDevicePlan&&) = delete;

    bool scheduled(Index tact) const;
    Index frame_size() const;
    void capture(
        DeviceExecutionContext& run, Real* output_frame, cudaStream_t stream) const;
    // If stream differs from run.stream(), runtime must make it wait on the
    // run's field-ready event before capture and keep output alive through D2H.

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct VelocityFrame {
    const Index* nodes;
    const Index* destinations;
    const Real* weights;
    const Code* components;
    Index entries, component_count, frame_size, n_count;
    const Real* solution;
    Real* frame;
};
cudaError_t gather_velocity_frame(const VelocityFrame&, cudaStream_t);
cudaError_t flag_receiver_frame(const Real*, Index, Code*, cudaStream_t);
}
