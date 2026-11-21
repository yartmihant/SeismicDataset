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

// Connectivity maps each collocated element point to a global node.
struct DeviceTopologyData {
    DeviceBuffer<Index> p_nodes;
    DeviceTopologyData(CleanupStatus& cleanup, const TenzorSpaceData& host, cudaStream_t stream)
        : p_nodes(cleanup, static_cast<Index>(host.p_nodes.size())) {
        p_nodes.upload(host.p_nodes.data(), stream);
    }
};
}  // namespace semgpu2_seismic_cu
