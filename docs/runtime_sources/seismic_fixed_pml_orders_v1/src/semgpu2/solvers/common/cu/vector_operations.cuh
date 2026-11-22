#pragma once

#include <cmath>
#include <cuda_runtime.h>

namespace semgpu2::common::cu {

// The caller supplies a nonnegative floor. Input/output must not alias.
// Matches common/nb/vector_operations.py::inverse_2d, including floor sign.
__device__ inline double inverse_2d(
    const double (&matrix)[2][2], double (&inverse)[2][2], double determinant_floor) {
    double determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if (fabs(determinant) < determinant_floor) {
        determinant = determinant >= 0.0 ? determinant_floor : -determinant_floor;
    }
    const double inverse_determinant = 1.0 / determinant;
    inverse[0][0] = matrix[1][1] * inverse_determinant;
    inverse[0][1] = -matrix[0][1] * inverse_determinant;
    inverse[1][0] = -matrix[1][0] * inverse_determinant;
    inverse[1][1] = matrix[0][0] * inverse_determinant;
    return determinant;
}

}  // namespace semgpu2::common::cu
