#pragma once
/**
 * @file Precision.hpp
 * @brief Type aliases for floating-point templated Eigen matrices.
 *
 * All modules in this project use PrecisionTraits<Scalar>::Matrix / Vector
 * instead of raw Eigen types.  This centralises precision switching:
 * change Scalar at the call site and the entire pipeline recompiles.
 */

#include <Eigen/Dense>

namespace randnla {

/**
 * @brief Traits class mapping a scalar type to its Eigen matrix/vector types.
 *
 * Supported Scalar: float, double, long double.
 */
template <typename Scalar>
struct PrecisionTraits {
    /// Dynamic-size matrix of the chosen scalar type.
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /// Dynamic-size column vector of the chosen scalar type.
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /// The scalar itself (useful in generic code).
    using ScalarType = Scalar;
};

// Convenience aliases
using FloatTraits  = PrecisionTraits<float>;
using DoubleTraits = PrecisionTraits<double>;
using LongTraits   = PrecisionTraits<long double>;

} // namespace randnla