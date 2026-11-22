#pragma once

#include "device_resources.hpp"

namespace SEMGPU2_SEISMIC_NAMESPACE {
// Connectivity maps each collocated element point to a global node.
struct DeviceTopologyData {
    DeviceBuffer<Index> p_nodes;
    DeviceTopologyData(CleanupStatus& cleanup, const TenzorSpaceData& host, cudaStream_t stream)
        : p_nodes(cleanup, static_cast<Index>(host.p_nodes.size())) {
        p_nodes.upload(host.p_nodes.data(), stream);
    }
};
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
