#include "receivers.hpp"
#include "stepping.hpp"
#include <algorithm>
#include <stdexcept>

namespace semgpu2_seismic_cu {
namespace {
struct StreamFence {
    CleanupStatus& cleanup; cudaStream_t stream; bool armed = true;
    ~StreamFence() { if (armed) cleanup.record(cudaStreamSynchronize(stream)); }
    void dismiss() { armed = false; }
};
} // namespace
    std::vector<Receiver> prepare_receivers(const std::vector<Receiver>& receivers, const CompiledState& state) {
        std::vector<Receiver> result;
        result.reserve(receivers.size());
        for (const Receiver& source : receivers) {
            const Code space = source.space_id;
            if (space != static_cast<Code>(SGSpaceEnum::N) && space != static_cast<Code>(SGSpaceEnum::MAIN) && space
                != static_cast<Code>(SGSpaceEnum::BOUNDARY)) throw
                    std::invalid_argument("receiver has invalid SpaceId");
            if (source.stride <= 0 || !((source.begin_tact == -1 && source.end_tact == -1) || (source.begin_tact >=
                0 && (source.end_tact == -1 || source.end_tact >= source.begin_tact)))) throw
                std::invalid_argument("invalid receiver schedule");
            if (source.frame_size < 0 || source.frame_buf_size < 0) throw
                std::invalid_argument("negative receiver frame size");
            (void)checked_product(source.frame_size, static_cast<Index>(source.comp_codes.size()));
            const Index entity_count = static_cast<Index>(source.sens_entity_ids.size());
            if (static_cast<Index>(source.sens_entity_offsets.size()) != entity_count + 1 ||
                source.sens_entity_offsets.empty() || source.sens_entity_offsets.front() != 0 ||
                source.sens_entity_offsets.back() != static_cast<Index>(source.sens_weights.size()) ||
                source.sens_frame_idxs.size() != source.sens_weights.size()) throw
                    std::invalid_argument("invalid receiver entity offsets");
            for (Index index = 1; index <= entity_count; ++index) if
                (source.sens_entity_offsets[static_cast<std::size_t>(index)] <
                source.sens_entity_offsets[static_cast<std::size_t>(index - 1)]) throw
                std::invalid_argument("nonmonotone receiver entity offsets");
            for (Code component : source.comp_codes) if (component != static_cast<Code>(SGReceiverComponentEnum::VX) && component != static_cast<Code>(SGReceiverComponentEnum::VY)) throw
                std::invalid_argument("receiver component is unavailable in its space");
            const TenzorSpaceData& carrier = space == static_cast<Code>(SGSpaceEnum::MAIN) ? state.main : state.boundary;
            for (Index entity = 0; entity < entity_count; ++entity) {
                const Index id = source.sens_entity_ids[static_cast<std::size_t>(entity)];
                const Index width = source.sens_entity_offsets[static_cast<std::size_t>(entity + 1)] -
                    source.sens_entity_offsets[static_cast<std::size_t>(entity)];
                if (space == static_cast<Code>(SGSpaceEnum::N)) {
                    if (id < 0 || id >= state.n_count || width != 1) throw
                        std::invalid_argument("N receiver entity is invalid");
                } else {
                    if (id < 0 || id >= carrier.element_count() || width !=
                        carrier.p_node_offsets[static_cast<std::size_t>(id + 1)] -
                        carrier.p_node_offsets[static_cast<std::size_t>(id)]) throw
                            std::invalid_argument("PQ receiver entity does not contain its complete P block");
                }
            }
            for (Index frame_index : source.sens_frame_idxs) if (frame_index < 0 || frame_index >=
                source.frame_size) throw std::invalid_argument("receiver frame index is outside frame");
            result.push_back(source);
        }
        return result;
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
            const auto& space = data.space_id == static_cast<Code>(SGSpaceEnum::MAIN) ? state.main : state.boundary;
            for (Index e : data.sens_entity_ids) {
                host_nodes.insert(host_nodes.end(), space.p_nodes.begin() + space.p_node_offsets[e],
                    space.p_nodes.begin() + space.p_node_offsets[e + 1]);
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
