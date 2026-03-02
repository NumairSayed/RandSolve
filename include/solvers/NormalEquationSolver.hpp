#pragma once
/**
 * @file NormalEquationSolver.hpp
 * @brief Least-squares via Normal Equations: solves (A^T A) x = A^T b.
 *
 * Faster than QR for dense tall matrices when n is small relative to m,
 * but squares the condition number.  Use only for well-conditioned A.
 *
 * Solver chain:
 *   1. Form G = A^T A  (n×n symmetric positive-definite)
 *   2. Form c = A^T b  (n vector)
 *   3. Solve G x = c   via Cholesky (LDLT for numerical stability)
 *
 * @tparam Scalar  float or double.
 */

#include "ILeastSquaresSolver.hpp"
#include <Eigen/Cholesky>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class NormalEquationSolver : public ILeastSquaresSolver<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    Vector solve(const Matrix& A, const Vector& b) override {
        if (A.rows() < A.cols())
            throw std::invalid_argument(
                "NormalEquationSolver: system must be overdetermined (m >= n)");
        // G = A^T A  (symmetric positive semi-definite)
        Matrix G = A.transpose() * A;
        Vector c = A.transpose() * b;
        // LDLT decomposition (robust Cholesky variant)
        return G.ldlt().solve(c);
    }
};

} // namespace randnla