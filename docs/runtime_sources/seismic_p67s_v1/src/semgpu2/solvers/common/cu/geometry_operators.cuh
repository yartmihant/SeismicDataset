#pragma once

#include "vector_operations.cuh"
#include <cstdint>

namespace semgpu2::common::cu {

// Port of the (physical_dim=2, rank=2) NB reducer. difgeoms is SoA
// [reference_direction, physical_direction, q]; the measure remains signed.
__device__ inline void reduce_full_rank_geometry(
    const double (&jacobian)[2][2], double* q_difgeoms, double* q_sizes,
    std::int64_t qid, std::int64_t q_count, double weight) {
    double inverse[2][2];
    const double determinant = inverse_2d(jacobian, inverse, 1.0e-14);
    q_sizes[qid] = determinant * weight;
    for (int reference = 0; reference < 2; ++reference) {
        for (int physical = 0; physical < 2; ++physical) {
            q_difgeoms[(reference * 2 + physical) * q_count + qid] = inverse[reference][physical];
        }
    }
}

// Port of the (physical_dim=2, rank=1) NB reducer. First row is the
// pseudoinverse, second row the oriented normal; publication belongs to caller.
template<int Columns>
__device__ inline void reduce_embedded_geometry(
    const double (&jacobian)[2][Columns], double* q_difgeoms, double* q_sizes,
    std::int64_t qid, std::int64_t q_count, double weight) {
    const double tangent_x = jacobian[0][0];
    const double tangent_y = jacobian[1][0];
    double measure = sqrt(tangent_x * tangent_x + tangent_y * tangent_y);
    if (measure < 1.0e-20) {
        measure = 1.0e-20;
    }
    const double inverse_measure_squared = 1.0 / (measure * measure);
    q_difgeoms[qid] = tangent_x * inverse_measure_squared;
    q_difgeoms[q_count + qid] = tangent_y * inverse_measure_squared;
    q_difgeoms[2 * q_count + qid] = -tangent_y / measure;
    q_difgeoms[3 * q_count + qid] = tangent_x / measure;
    q_sizes[qid] = measure * weight;
}
}  // namespace semgpu2::common::cu

namespace semgpu2::common::cu {
// Rank-three consumers use scale-invariant admission before any division.
// Embedded storage has two dual-tangent rows followed by the unit normal.
template<int Rank, int Columns>
__device__ inline void reduce_scaled_geometry_3d(
    const double (&jacobian)[3][Columns], double* mapping, double* sizes,
    std::int64_t qid, std::int64_t q_count, double weight, double* tangents = nullptr) {
    static_assert(Rank == 2 || Rank == 3);
    static_assert(Columns >= Rank);
    double scaled[3][3]{}, scale[3]{}, length[3]{};
    bool valid = isfinite(weight) && weight > 0.0;
    for (int a = 0; a < Rank; ++a) {
        for (int d = 0; d < 3; ++d) {
            valid = valid && isfinite(jacobian[d][a]);
            scale[a] = fmax(scale[a], fabs(jacobian[d][a]));
        }
        valid = valid && scale[a] > 0.0 && isfinite(scale[a]);
    }
    if (valid) {
        for (int a = 0; a < Rank; ++a) {
            for (int d = 0; d < 3; ++d) {
                scaled[d][a] = jacobian[d][a] / scale[a];
                length[a] += scaled[d][a] * scaled[d][a];
            }
            length[a] = sqrt(length[a]);
        }
        double cross[3];
        for (int d = 0; d < 3; ++d) {
            const int b = (d + 1) % 3, c = (d + 2) % 3;
            cross[d] = scaled[b][0] * scaled[c][1] - scaled[c][0] * scaled[b][1];
        }
        if constexpr (Rank == 3) {
            double determinant = 0.0;
            for (int d = 0; d < 3; ++d) determinant += cross[d] * scaled[d][2];
            valid = determinant > 1e-12 * length[0] * length[1] * length[2];
            if (valid) {
                for (int a = 0; a < 3; ++a) for (int d = 0; d < 3; ++d) {
                    const int b = (a + 1) % 3, c = (a + 2) % 3;
                    const int e = (d + 1) % 3, f = (d + 2) % 3;
                    mapping[(3 * a + d) * q_count + qid] =
                        ((scaled[e][b] * scaled[f][c] - scaled[f][b] * scaled[e][c]) / determinant) / scale[a];
                }
                sizes[qid] = exp(log(weight) + log(determinant) + log(scale[0]) + log(scale[1]) + log(scale[2]));
            }
        } else {
            const double area = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
            valid = area > 1e-12 * length[0] * length[1];
            if (valid) {
                for (int d = 0; d < 3; ++d) {
                    const int b = (d + 1) % 3, c = (d + 2) % 3;
                    mapping[d * q_count + qid] = ((scaled[b][1] * cross[c] - scaled[c][1] * cross[b]) / (area * area)) / scale[0];
                    mapping[(3 + d) * q_count + qid] = ((cross[b] * scaled[c][0] - cross[c] * scaled[b][0]) / (area * area)) / scale[1];
                    mapping[(6 + d) * q_count + qid] = cross[d] / area;
                    if (tangents) tangents[d * q_count + qid] = scaled[d][0] / length[0];
                }
                sizes[qid] = exp(log(weight) + log(area) + log(scale[0]) + log(scale[1]));
            }
        }
    }
    if (!valid) {
        sizes[qid] = nan("");
        for (int row = 0; row < 9; ++row) mapping[row * q_count + qid] = nan("");
        if (tangents) for (int d = 0; d < 3; ++d) tangents[d * q_count + qid] = nan("");
    }
}
} // namespace semgpu2::common::cu
