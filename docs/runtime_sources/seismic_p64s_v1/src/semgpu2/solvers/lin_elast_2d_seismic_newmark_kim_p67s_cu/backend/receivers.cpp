#include "receivers.hpp"
#include "stepping.hpp"
#include <algorithm>
#include <stdexcept>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
struct StreamFence {
    CleanupStatus& cleanup; cudaStream_t stream; bool armed = true;
    ~StreamFence() { if (armed) cleanup.record(cudaStreamSynchronize(stream)); }
    void dismiss() { armed = false; }
};
} // namespace
std::vector<Receiver> prepare_receivers(std::vector<Receiver> receivers, const CompiledState& state) {
    for (const Receiver& receiver : receivers) {
        const bool nodal = receiver.space_id == static_cast<Code>(SGSpaceEnum::N);
        if (!nodal && receiver.space_id != static_cast<Code>(SGSpaceEnum::MAIN))
            throw std::invalid_argument("Receiver space must be N or MAIN");
        const bool terminal_only = receiver.begin_tact == -1 && receiver.end_tact == -1;
        const bool scheduled = receiver.begin_tact >= 0 &&
            (receiver.end_tact == -1 || receiver.end_tact >= receiver.begin_tact);
        if (receiver.stride <= 0 || !(scheduled || terminal_only))
            throw std::invalid_argument("Invalid receiver schedule");
        if (receiver.frame_size < 0 || receiver.frame_buf_size < 0)
            throw std::invalid_argument("Negative receiver frame size");
        checked_product(receiver.frame_size, static_cast<Index>(receiver.comp_codes.size()));

        for (Code code : receiver.comp_codes) {
            if (code < 0 || code / 6 >= SHOT_COUNT || code % 6 < static_cast<Code>(SGReceiverComponentEnum::VX))
                throw std::invalid_argument("Receiver component must select a shot velocity");
        }
        const Index width = nodal ? 1 : state.main.nodes_per_element;
        const Index count = nodal ? state.n_count : state.main.element_count();
        for (Index entity : receiver.sens_entity_ids) {
            if (entity < 0 || entity >= count)
                throw std::invalid_argument("Receiver entity is outside its space");
        }
        const Index entries = checked_product(width, static_cast<Index>(receiver.sens_entity_ids.size()));
        if (entries != static_cast<Index>(receiver.sens_weights.size()) ||
            receiver.sens_frame_idxs.size() != receiver.sens_weights.size())
            throw std::invalid_argument("Receiver weights and frame indices must cover every local position");
        for (Index index : receiver.sens_frame_idxs) {
            if (index < 0 || index >= receiver.frame_size)
                throw std::invalid_argument("Receiver frame index is outside frame");
        }
    }
    return receivers;
}

struct ReceiverDevicePlan::Impl {
    CleanupStatus& cleanup;
    Receiver host;
    std::vector<Index> host_nodes;
    DeviceBuffer<Index> nodes, destinations;
    DeviceBuffer<Real> weights;
    DeviceBuffer<Code> components, nonfinite;

    Impl(CleanupStatus& status, const Receiver& data, const CompiledState& state, cudaStream_t stream)
        : cleanup(status), host(data), nodes(status, data.sens_weights.size()),
          destinations(status, data.sens_frame_idxs.size()), weights(status, data.sens_weights.size()),
          components(status, data.comp_codes.size()), nonfinite(status, 1) {
        // Expand the receiver's ordered P occurrences onto global N once. Duplicate
        // nodes and destinations remain independent terms of the sparse frame.
        if (data.space_id == static_cast<Code>(SGSpaceEnum::N)) {
            host_nodes = data.sens_entity_ids;
        } else {
            const auto& space = state.main;
            for (Index e : data.sens_entity_ids) {
                host_nodes.insert(host_nodes.end(), space.p_nodes.begin() + e * space.nodes_per_element,
                    space.p_nodes.begin() + (e + 1) * space.nodes_per_element);
            }
        }
        StreamFence fence{cleanup, stream};
        nodes.upload(host_nodes.data(), stream);
        destinations.upload(host.sens_frame_idxs.data(), stream);
        weights.upload(host.sens_weights.data(), stream);
        components.upload(host.comp_codes.data(), stream);
        cuda_check(cudaStreamSynchronize(stream), "upload velocity receiver");
        fence.dismiss();
    }
};
ReceiverDevicePlan::ReceiverDevicePlan(CleanupStatus& cleanup, const Receiver& host,
    const CompiledState& state, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(cleanup, host, state, stream)) {}
ReceiverDevicePlan::~ReceiverDevicePlan() = default;
bool ReceiverDevicePlan::scheduled(Index tact) const {
    const auto& h=impl_->host;
    return h.begin_tact>=0 && tact>=h.begin_tact && (h.end_tact<0 || tact<=h.end_tact) && (tact-h.begin_tact)%h.stride==0;
}
Index ReceiverDevicePlan::frame_size() const { return impl_->host.frame_size; }
void ReceiverDevicePlan::capture(DeviceExecutionContext& run, Real* output, cudaStream_t stream) const {
    const auto& h = impl_->host;
    const Index count = checked_product(static_cast<Index>(h.comp_codes.size()), h.frame_size);
    if (!count) return;
    Code flag = 0;
    StreamFence fence{impl_->cleanup, stream};
    cuda_check(cudaMemsetAsync(output, 0, array_size<Real>(count) * sizeof(Real), stream), "clear velocity frame");
    impl_->nonfinite.clear(stream);
    const VelocityFrame frame{impl_->nodes.get(), impl_->destinations.get(), impl_->weights.get(),
        impl_->components.get(), static_cast<Index>(h.sens_weights.size()), static_cast<Index>(h.comp_codes.size()),
        h.frame_size, run.n_count(), run.nodal_solution(), output};
    cuda_check(gather_velocity_frame(frame, stream), "gather velocity frame");
    cuda_check(flag_receiver_frame(output, count, impl_->nonfinite.get(), stream), "check velocity frame");
    impl_->nonfinite.download(&flag, stream);
    cuda_check(cudaStreamSynchronize(stream), "finish velocity frame");
    fence.dismiss();
    if (flag) throw NumericalFailure("nonfinite velocity frame");
}
}
