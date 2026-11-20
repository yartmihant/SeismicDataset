#include "stepping.hpp"
namespace semgpu2_seismic_cu {
    namespace {
        ResponseLaunchData launch_data(DeviceExecutionContext& run, const PreparedField& field, DevicePreparedField& device) {
            if (!device.has_schedule) throw std::logic_error("PQ field requires its own packed schedule");
            return {
                run.space(static_cast<SGSpaceEnum>(field.space_id)), run.n_count(),
                device.entity_ids.get(), device.node_offsets.get(), device.values.get(),
                field.node_count(),
                device.thread_slots.get(), device.slot_entity_indices.get(), device.slot_thread_offsets.get(),
                field.schedule.slots_per_block, field.schedule.block_count
            };
        }
        void checked(cudaError_t error, const char* context) {
            cuda_check(error, context);
        }
    }
    // namespace
    DevicePreparedField::DevicePreparedField(CleanupStatus& cleanup, const PreparedField& source, cudaStream_t stream)
    : host(&source), entity_ids(cleanup, static_cast<Index>(source.entity_ids.size())),
    node_offsets(cleanup, static_cast<Index>(source.node_offsets.size())),
    values(cleanup, source.layer_size()),
    thread_slots(cleanup, source.has_schedule ? static_cast<Index>(source.schedule.thread_slots.size()) : 0),
    slot_entity_indices(cleanup, source.has_schedule ? static_cast<Index>(source.schedule.slot_entity_indices.size()) : 0),
    slot_thread_offsets(cleanup, source.has_schedule ? static_cast<Index>(source.schedule.slot_thread_offsets.size()) : 0),
    has_schedule(source.has_schedule) {
        try {
            entity_ids.upload(source.entity_ids.data(), stream);
            node_offsets.upload(source.node_offsets.data(), stream);
            if (has_schedule) {
                thread_slots.upload(source.schedule.thread_slots.data(), stream);
                slot_entity_indices.upload(source.schedule.slot_entity_indices.data(), stream);
                slot_thread_offsets.upload(source.schedule.slot_thread_offsets.data(), stream);
            }
            if (!source.dynamic) values.upload(source.layer(), stream);
        } catch (...) {
            // Queued H2D copies retain host source/evaluated storage until done.
            cleanup.record(cudaStreamSynchronize(stream));
            throw;
        }
    }
    void DevicePreparedField::upload_at(Real time, cudaStream_t stream) {
        if (!host->dynamic) return;
        evaluated = evaluate_physics_values(*host, time);
        values.upload(evaluated.data(), stream);
        // The next evaluation may replace this host storage before the end of the tact.
        cuda_check(cudaStreamSynchronize(stream), "complete dynamic force upload");
    }
    DeviceExecutionContext::DeviceExecutionContext(CleanupStatus& cleanup, const CompiledState& state,
        const std::vector<Real>& nodal_mass, const std::vector<Real>& q_materials, const std::map<Index, PreparedField>& fields, bool check_finite)
    : cleanup_(cleanup), state_(state), stream_(cleanup), pets_(cleanup, state.pets, stream()),
    main_(cleanup, state.main, state.main_schedule, stream()), boundary_(cleanup, state.boundary, stream()),
    nodal_solution_(cleanup, checked_product(6, state.n_count)), nodal_response_(cleanup, checked_product(2, state.n_count)),
    nodal_mass_(cleanup, state.n_count), q_materials_(cleanup, checked_product(3, state.main.q_count())), q_main_geometry_(cleanup, checked_product(5, state.main.q_count())),
    q_boundary_geometry_(cleanup, checked_product(3, state.boundary.q_count())),
    nonfinite_flag_(cleanup, check_finite ? 1 : 0) {
        try {
            if (!state.meshed) throw std::logic_error("DeviceExecutionContext requires a meshed structure");
            nodal_solution_.upload(state.solution[static_cast<int>(SGSpaceEnum::N)].values.data(), stream());
            nodal_response_.clear(stream());
            q_materials_.upload(q_materials.data(), stream());
            if (static_cast<Index>(nodal_mass.size()) != state.n_count) throw std::logic_error("Prepared nodal mass shape mismatch");
            nodal_mass_.upload(nodal_mass.data(), stream());
            q_main_geometry_.upload(state.geometry[static_cast<int>(SGSpaceEnum::MAIN)].values.data(), stream());
            q_boundary_geometry_.upload(state.geometry[static_cast<int>(SGSpaceEnum::BOUNDARY)].values.data(), stream());
            for (const auto& [id, field] : fields) {
                primary_.emplace(id, std::make_unique<DevicePreparedField>(cleanup, field, stream()));
            }
            synchronize();
        } catch (...) {
            // DevicePreparedField::evaluated must outlive all queued uploads.
            cleanup_.record(cudaStreamSynchronize(stream()));
            throw;
        }
    }
    DeviceExecutionContext::~DeviceExecutionContext() noexcept {
        // Member destruction releases uploads and device buffers.  Synchronize
        // first so DevicePreparedField::evaluated remains valid through all H2D.
        cleanup_.record(cudaStreamSynchronize(stream_.get()));
    }
    DeviceSpaceView DeviceExecutionContext::space(SGSpaceEnum id) const {
        const bool main = id == SGSpaceEnum::MAIN;
        const auto& host = main ? state_.main : state_.boundary;
        const auto& device = main ? main_ : boundary_;
        return {
            pets_.view(), device.p_nodes.get(),
            device.p_node_offsets.get(), device.q_node_offsets.get(), host.q_count()
        };
    }
    void DeviceExecutionContext::synchronize() {
        stream_.synchronize();
    }
    void DeviceExecutionContext::copy_solution_to_host(Field& target) const {
        if (target.rows != 6 || target.columns != state_.n_count) throw std::invalid_argument("Solution host field shape mismatch");
        nodal_solution_.download(target.values.data(), stream());
        stream_.synchronize();
    }
    void DeviceExecutionContext::assemble_force_response(Real time, std::optional<Code> category) {
        nodal_response_.clear(stream());
        if (!category || *category == 1) {
            const auto& schedule = state_.main_schedule;
            const ResponseLaunchData data{space(SGSpaceEnum::MAIN), n_count(),
                nullptr, nullptr, q_materials_.get(), state_.main.q_count(),
                main_.thread_slots.get(), main_.slot_entity_indices.get(), main_.slot_thread_offsets.get(),
                schedule.slots_per_block, schedule.block_count};
            checked(assemble_material_response(data, nodal_solution(), q_main_measures(),
                q_main_inv_jacobians(), nodal_response(), stream()), "internal force");
        }
        for (const auto& [id, device] : primary_) {
            const auto& field = *device->host;
            if (category && field.type_code / 10000 != *category) continue;
            switch (static_cast<SGPhysicsTypeEnum>(field.type_code)) {
            case SGPhysicsTypeEnum::N_FORCE_DYNAMIC:
                device->upload_at(time, stream());
                checked(assemble_nodal_force(*device, device->values.get(), n_count(),
                    nodal_response(), stream()), "dynamic nodal force");
                break;
            case SGPhysicsTypeEnum::N_MASS_DAMPING_STATIC:
                checked(assemble_nodal_damping(*device, device->values.get(), n_count(), nodal_mass(),
                    nodal_solution(), nodal_response(), stream()), "mass damping");
                break;
            case SGPhysicsTypeEnum::ABC:
                checked(assemble_boundary_damping(launch_data(*this, field, *device), nodal_solution(),
                    q_boundary_measures(), q_boundary_normals(), nodal_response(), stream()), "ABC");
                break;
            default: throw std::logic_error("Unexpected prepared physics field");
            }
        }
    }
    bool DeviceExecutionContext::check_finite_solution() {
        Code flag = 0;
        try {
            checked(cudaMemsetAsync(nonfinite_flag_.get(), 0, sizeof(Code), stream()), "clear finite flag");
            checked(flag_nonfinite_solution(nodal_solution(), nodal_response(), n_count(), nonfinite_flag_.get(), stream()), "finite check");
            nonfinite_flag_.download(&flag, stream());
            checked(cudaStreamSynchronize(stream()), "synchronize finite check");
        } catch (...) {
            // Keep the D2H target alive until every queued copy has stopped.
            cleanup_.record(cudaStreamSynchronize(stream()));
            throw;
        }
        return flag == 0;
    }
    bool DeviceExecutionContext::advance_internal_step(Real time, Real dt, Real beta, Real gamma, bool check_finite) {
        checked(predict_nodal_state(nodal_solution(), n_count(), dt, stream()), "predict Newmark state");
        assemble_force_response(time);
        checked(correct_nodal_state(nodal_solution(), nodal_response(), nodal_mass(), n_count(), dt, beta, gamma, stream()), "correct Newmark state");
        return !check_finite || check_finite_solution();
    }
}  // namespace semgpu2_seismic_cu
