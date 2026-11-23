#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semgpu2::common {

// Rows contain d, a, b, and in 3D the integral weight c, followed by the
// independent Voigt stiffness entries. Accumulate the constitutive tensors
// of the actual material occurrences, including overlaps.
template<int D, class Field, class Materials, class Law>
void prepare_pml(Field& field, const Materials& materials, double step, Law law) {
    constexpr int voigt = D*(D+1)/2;
    constexpr int stiffness = D == 2 ? 6 : 12;
    const auto count = field.node_count();
    field.component_count = stiffness + voigt*(voigt+1)/2;
    field.values.resize(field.layer_size());
    for (int j = 0; j < D; ++j) {
        for (decltype(field.node_count()) q = 0; q < count; ++q) {
            const double d = field.values[j*count+q];
            if (!std::isfinite(d) || d < 0) throw std::invalid_argument("Invalid summed PML stretch");
            const double z = d*step, a = std::exp(-z);
            const double b = a == 1.0 ? step : -std::expm1(-z)/d;
            field.values[(D+j)*count+q] = a;
            field.values[(2*D+j)*count+q] = b;
            if constexpr (D == 3) {
                field.values[(3*D+j)*count+q] = z < 1e-3
                    ? step*(step*(.5+z*(-1./6+z*(1./24-z/120))))
                    : (step-b)/d;
            }
        }
    }
    for (const auto& [code, material] : materials) {
        if (code / 10000 != 1) continue;
        for (std::size_t i = 0; i < material.entity_ids.size(); ++i) {
            const auto id = material.entity_ids[i];
            const auto found = std::lower_bound(field.entity_ids.begin(), field.entity_ids.end(), id);
            if (found == field.entity_ids.end() || *found != id) continue;
            const auto begin = material.node_offsets[i], end = material.node_offsets[i+1];
            const auto dst = field.node_offsets[found-field.entity_ids.begin()];
            for (auto q = begin; q < end; ++q) {
                int row = stiffness;
                for (int col = 0; col < voigt; ++col) {
                    double unit[voigt]{}, stress[voigt]{};
                    unit[col] = 1;
                    law((code/10)%10,material.values.data(),material.node_count(),q,unit,stress);
                    for (int k = col; k < voigt; ++k) field.values[row++*count+dst+q-begin] += stress[k];
                }
            }
        }
    }
    for (double value : field.values)
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite prepared PML coefficient");
}

} // namespace semgpu2::common
