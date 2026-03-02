#pragma once
/**
 * @file ILeastSquaresSolver.hpp
 * @brief Abstract interface for deterministic least-squares solvers.
 *
 * Solvers in this hierarchy:
 *   - Have NO dependency on sketching or RNG.
 *   - Accept any matrix/vector of the appropriate Scalar type.
 *   - May assume the system is overdetermined (m >= n).
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"

namespace randnla {

template <typename Scalar>
class ILeastSquaresSolver {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    virtual ~ILeastSquaresSolver() = default;

    /**
     * @brief Solve min_x ||Ax - b||_2.
     * @param A  System matrix, shape (m, n), m >= n.
     * @param b  Right-hand side, length m.
     * @return   Solution vector x, length n.
     */
    virtual Vector solve(const Matrix& A, const Vector& b) = 0;

    /**
     * @brief Compute the residual norm ||Ax - b||_2 for a given solution.
     *
     * Default implementation; subclasses may override with a more
     * efficient variant (e.g., if the decomposition already computed it).
     */
    virtual Scalar residual_norm(const Matrix& A, const Vector& b, const Vector& x) {
        return (A * x - b).norm();
    }
};

} // namespace randnla