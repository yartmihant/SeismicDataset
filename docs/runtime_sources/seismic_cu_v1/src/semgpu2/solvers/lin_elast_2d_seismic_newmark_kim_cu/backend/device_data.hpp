#pragma once

#include "device_resources.hpp"

namespace semgpu2_seismic_cu {
// Operation-local uploads. The host coefficient owner outlives queued transfers.
struct DevicePETData {
    Index node_count;
    DeviceBuffer<double> pq_grads, q_weights;
    DevicePETData(CleanupStatus& cleanup, const TenzorPETData& host, cudaStream_t stream)
        : node_count(host.node_count),
          pq_grads(cleanup, static_cast<Index>(host.pq_grads.size())),
          q_weights(cleanup, static_cast<Index>(host.q_weights.size())) {
        pq_grads.upload(host.pq_grads.data(), stream);
        q_weights.upload(host.q_weights.data(), stream);
    }
    PETView view() const { return {node_count, pq_grads.get(), q_weights.get()}; }
};

// Runtime owns only topology. Geometry preparation adds its own packed schedule.
struct DeviceTopologyData {
    DeviceBuffer<Index> p_nodes, p_node_offsets, q_node_offsets;
    DeviceTopologyData(CleanupStatus& cleanup, const TenzorSpaceData& host, cudaStream_t stream)
        : p_nodes(cleanup, static_cast<Index>(host.p_nodes.size())),
          p_node_offsets(cleanup, static_cast<Index>(host.p_node_offsets.size())),
          q_node_offsets(cleanup, static_cast<Index>(host.q_node_offsets.size())) {


        p_nodes.upload(host.p_nodes.data(), stream);
        p_node_offsets.upload(host.p_node_offsets.data(), stream);
        q_node_offsets.upload(host.q_node_offsets.data(), stream);
    }
};
struct DeviceSpaceData : DeviceTopologyData {
    DeviceBuffer<Index> thread_slots, slot_entity_indices, slot_thread_offsets;
    DeviceSpaceData(CleanupStatus& cleanup, const TenzorSpaceData& host,
        const BlockSchedule& schedule, cudaStream_t stream)
        : DeviceTopologyData(cleanup, host, stream),
          thread_slots(cleanup, static_cast<Index>(schedule.thread_slots.size())),
          slot_entity_indices(cleanup, static_cast<Index>(schedule.slot_entity_indices.size())),
          slot_thread_offsets(cleanup, static_cast<Index>(schedule.slot_thread_offsets.size())) {
        thread_slots.upload(schedule.thread_slots.data(), stream);
        slot_entity_indices.upload(schedule.slot_entity_indices.data(), stream);
        slot_thread_offsets.upload(schedule.slot_thread_offsets.data(), stream);
    }
};
}  // namespace semgpu2_seismic_cu
