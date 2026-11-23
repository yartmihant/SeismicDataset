#include "stepping.hpp"

#include <algorithm>
#include <cmath>

namespace SEMGPU2_SEISMIC_NAMESPACE {
    namespace {
        // Gather the three canonical elastic rows in material-carrier order.
        // A PML material prefixes them with its six prepared recurrence rows.
        void pack_elastic_rows(const TenzorSpaceData& space, const std::vector<Index>& elements,
            const std::vector<Real>& material, Index first_row, std::vector<Real>& packed) {
            const Index width = space.nodes_per_element;
            const Index stride = checked_product(static_cast<Index>(elements.size()), width);
            for (Index row = 0; row < 3; ++row) {
                Index dest = (first_row + row) * stride;
                for (Index element : elements) {
                    const Index src = row * space.q_count() + element * width;
                    std::copy_n(material.data() + src, width, packed.data() + dest);
                    dest += width;
                }
            }
        }
    }  // namespace

    DeviceExecutionContext::DeviceExecutionContext(CleanupStatus& cleanup, const CompiledState& state,
        const std::vector<Real>& nodal_mass, const std::vector<Real>& q_materials, const std::vector<Real>& source_samples,
        const std::optional<PreparedField>& pml_field, const std::vector<Real>& pml_history, bool check_finite,
        Index shot_count)
    : cleanup_(cleanup), state_(state), shot_count_(shot_count), use_inverse_mass_(false), stream_(cleanup),
    main_(cleanup, state.main, stream()),
    pml_q_count_(pml_field ? pml_field->node_count() : 0),
    ordinary_entity_count_(state.main.element_count() - (pml_field ? static_cast<Index>(pml_field->entity_ids.size()) : 0)),
    nodal_solution_(cleanup, checked_product(6 * shot_count, state.n_count)),
    nodal_response_(cleanup, checked_product(2 * shot_count, state.n_count)),
    nodal_mass_(cleanup, state.n_count),
    ordinary_materials_(cleanup, checked_product(3, state.main.q_count() - pml_q_count_)),
    q_main_geometry_(cleanup, checked_product(5, state.main.q_count())),
    pml_history_(cleanup, static_cast<Index>(pml_history.size())),
    pml_materials_(cleanup, checked_product(pml::ROW_COUNT + 3, pml_q_count_)),
    ordinary_entity_ids_(cleanup, pml_field ? ordinary_entity_count_ : 0),
    pml_entity_ids_(cleanup, pml_field ? static_cast<Index>(pml_field->entity_ids.size()) : 0),
    nonfinite_flag_(cleanup, check_finite ? 1 : 0),
    source_samples_(cleanup, static_cast<Index>(source_samples.size())) {
        std::vector<Real> fused_material;
        std::vector<Real> ordinary_material;
        std::vector<Index> ordinary;
        std::vector<Real> inverse_mass;
        try {
            if (!state.meshed) throw std::logic_error("DeviceExecutionContext requires a meshed structure");
            if (shot_count_ != 1 && shot_count_ != SHOT_COUNT) throw std::logic_error("Invalid shot count");
            if (shot_count_ == 1 && !source_samples.empty())
                throw std::logic_error("CFL probe cannot receive source samples");
            const auto& solution = state.solution[static_cast<int>(SGSpaceEnum::N)];
            if (solution.rows != 6 * SHOT_COUNT || solution.columns != state.n_count)
                throw std::logic_error("Shot-major solution shape mismatch");
            if (shot_count_ == 1) nodal_solution_.clear(stream());
            else nodal_solution_.upload(solution.values.data(), stream());
            nodal_response_.clear(stream());
            if (static_cast<Index>(nodal_mass.size()) != state.n_count) throw std::logic_error("Prepared nodal mass shape mismatch");
            if (pml_field && (pml_field->component_count != pml::ROW_COUNT
                || pml_field->node_count() != checked_product(static_cast<Index>(pml_field->entity_ids.size()), state.main.nodes_per_element)))
                throw std::logic_error("Prepared PML field shape mismatch");
            if (pml_history_.size() != checked_product(checked_product(4, shot_count_), pml_field ? pml_field->node_count() : 0))
                throw std::logic_error("PML history shape mismatch");
            const Real* run_mass = nodal_mass.data();
            if (shot_count_ == SHOT_COUNT) {
                inverse_mass.reserve(nodal_mass.size());
                use_inverse_mass_ = true;
                for (const Real mass : nodal_mass) {
                    const Real inverse = mass > 0.0 ? 1.0 / mass : 0.0;
                    if (!(inverse > 0.0) || !std::isfinite(inverse)) {
                        use_inverse_mass_ = false;
                        break;
                    }
                    inverse_mass.push_back(inverse);
                }
                if (use_inverse_mass_) run_mass = inverse_mass.data();
            }
            nodal_mass_.upload(run_mass, stream());
            q_main_geometry_.upload(state.geometry[static_cast<int>(SGSpaceEnum::MAIN)].values.data(), stream());
            if (pml_field) {
                ordinary.reserve(array_size<Index>(ordinary_entity_count_));
                auto next_pml = pml_field->entity_ids.begin();
                for (Index element = 0; element < state.main.element_count(); ++element) {
                    if (next_pml != pml_field->entity_ids.end() && *next_pml == element) ++next_pml;
                    else ordinary.push_back(element);
                }
                ordinary_entity_ids_.upload(ordinary.data(), stream());
                ordinary_material.resize(array_size<Real>(ordinary_materials_.size()));
                pack_elastic_rows(state.main, ordinary, q_materials, 0, ordinary_material);
                ordinary_materials_.upload(ordinary_material.data(), stream());

                fused_material.resize(array_size<Real>(pml_materials_.size()));
                std::copy(pml_field->values.begin(), pml_field->values.end(), fused_material.begin());
                pack_elastic_rows(state.main, pml_field->entity_ids, q_materials, pml::ROW_COUNT, fused_material);
                pml_materials_.upload(fused_material.data(), stream());
                pml_entity_ids_.upload(pml_field->entity_ids.data(), stream());
            } else {
                ordinary_materials_.upload(q_materials.data(), stream());
            }
            pml_history_.upload(pml_history.data(), stream());
            for (Index shot = 0; shot < SHOT_COUNT; ++shot) {
                const Index node = source_profile::node(static_cast<int>(shot));
                if (shot_count_ == SHOT_COUNT && (node < 0 || node >= state.n_count)) {
                    throw std::logic_error(SEMGPU2_SEISMIC_SOLVER_ID " source node is outside the frozen N layout");
                }
            }
            source_samples_.upload(source_samples.data(), stream());
            synchronize();
        } catch (...) {
            // Temporary packed material rows must outlive all queued H2D copies.
            cleanup_.record(cudaStreamSynchronize(stream()));
            throw;
        }
    }
    DeviceExecutionContext::~DeviceExecutionContext() noexcept {
        // Member destruction releases device resources after outstanding uploads.
        cleanup_.record(cudaStreamSynchronize(stream_.get()));
    }
    void DeviceExecutionContext::synchronize() {
        stream_.synchronize();
    }
    void DeviceExecutionContext::copy_solution_to_host(Field& target) const {
        if (target.rows != checked_product(6, shot_count_) || target.columns != state_.n_count) throw std::invalid_argument("Solution host field shape mismatch");
        nodal_solution_.download(target.values.data(), stream());
        stream_.synchronize();
    }
    void DeviceExecutionContext::copy_pml_history_to_host(std::vector<Real>& target) const {
        if (static_cast<Index>(target.size()) != pml_history_.size()) throw std::invalid_argument("PML history host field shape mismatch");
        pml_history_.download(target.data(), stream());
        stream_.synchronize();
    }
    void DeviceExecutionContext::assemble_stiffness_probe() {
        nodal_response_.clear(stream());
        const ResponseLaunchData data{main_space(), n_count(), nullptr, ordinary_materials_.get(), state_.main.q_count(), state_.main.element_count()};
        cuda_check(assemble_material_response(data, nodal_solution(), q_main_measures(), q_main_inv_jacobians(), nodal_response(), shot_count_, stream()), "stiffness probe");
    }
    void DeviceExecutionContext::assemble_step_response(Index step_index, Real pml_dt) {
        nodal_response_.clear(stream());
        const ResponseLaunchData ordinary{main_space(), n_count(), ordinary_entity_ids_.get(), ordinary_materials_.get(), checked_product(ordinary_entity_count_, state_.main.nodes_per_element), ordinary_entity_count_};
        cuda_check(assemble_material_response(ordinary, nodal_solution(), q_main_measures(), q_main_inv_jacobians(), nodal_response(), shot_count_, stream()), "ordinary internal force");
        if (pml_q_count_) {
            const ResponseLaunchData pml{main_space(), n_count(), pml_entity_ids_.get(), pml_materials_.get(), pml_q_count_, static_cast<Index>(pml_entity_ids_.size())};
            cuda_check(assemble_pml_response(pml, nodal_solution(), q_main_measures(), q_main_inv_jacobians(), pml_history_.get(), pml_dt, nodal_response(), shot_count_, stream()), "fused PML response");
        }
        if (step_index < 0 || step_index >= source_samples_.size())
            throw std::logic_error("Source sample outside prepared time grid");
        cuda_check(assemble_prepared_source(source_samples_.get() + step_index, n_count(), nodal_response(), stream()),
            "prepared Ricker source");
    }
    bool DeviceExecutionContext::check_finite_solution() {
        Code flag = 0;
        try {
            cuda_check(cudaMemsetAsync(nonfinite_flag_.get(), 0, sizeof(Code), stream()), "clear finite flag");
            cuda_check(flag_nonfinite_solution(nodal_solution(), nodal_response(), n_count(), shot_count_, nonfinite_flag_.get(), stream()), "finite check");
            cuda_check(flag_nonfinite_values(pml_history_.get(), pml_history_.size(), nonfinite_flag_.get(), stream()), "PML finite check");
            nonfinite_flag_.download(&flag, stream());
            cuda_check(cudaStreamSynchronize(stream()), "synchronize finite check");
        } catch (...) {
            // Keep the D2H target alive until every queued copy has stopped.
            cleanup_.record(cudaStreamSynchronize(stream()));
            throw;
        }
        return flag == 0;
    }
    bool DeviceExecutionContext::advance_internal_step(Index step_index, Real dt, Real beta, Real gamma,
        bool check_finite, bool starts_predicted, bool predicts_next) {
        if (check_finite && predicts_next) throw std::logic_error("Finite checking requires an observable corrected state");
        if (!starts_predicted)
            cuda_check(predict_nodal_state(nodal_solution(), n_count(), dt, shot_count_, stream()), "predict Newmark state");
        assemble_step_response(step_index, dt);
        const auto correct = predicts_next ? correct_nodal_state<true> : correct_nodal_state<false>;
        cuda_check(correct(nodal_solution(), nodal_response(), nodal_mass(), use_inverse_mass_, n_count(),
            dt, beta, gamma, shot_count_, stream()), "correct Newmark state");
        return !check_finite || check_finite_solution();
    }
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
