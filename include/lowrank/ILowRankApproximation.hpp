#pragma once
/**
 * @file ILowRankApproximation.hpp
 * @brief Abstract interface for low-rank matrix approximation.
 *
 * A rank-k approximation produces:
 *   A ≈ U Σ V^T
 * where U ∈ R^{m×k}, Σ ∈ R^{k×k}, V ∈ R^{n×k}.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"

namespace randnla {

template <typename Scalar>
class ILowRankApproximation {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    virtual ~ILowRankApproximation() = default;

    /**
     * @brief Compute rank-k approximation of A.
     * @param A     Input matrix.
     * @param rank  Target rank k.
     */
    virtual void compute(const Matrix& A, int rank) = 0;

    /// Left singular vectors U, shape (m, k).
    virtual Matrix matrixU() const = 0;

    /// Singular values, length k.
    virtual Vector singularValues() const = 0;

    /// Right singular vectors transposed V^T, shape (k, n).
    virtual Matrix matrixVt() const = 0;

    /// Frobenius-norm reconstruction error ||A - U Σ V^T||_F.
    virtual Scalar reconstruction_error(const Matrix& A) const {
    Matrix Ak = matrixU() * singularValues().asDiagonal() * matrixVt();
    return (A - Ak).norm();
}
};

} // namespace randnla