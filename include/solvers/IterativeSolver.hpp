#pragma once
/**
 * @file IterativeSolver.hpp
 * @brief Iterative least-squares solver — Conjugate Gradient on Normal Equations.
 *
 * Solves min_x ||Ax - b||_2  by applying Conjugate Gradient to the normal
 * equations: (A^T A) x = A^T b.
 *
 * Suitable for large sparse systems where forming QR is prohibitive.
 *
 * Note: Eigen's built-in ConjugateGradient requires a SPD matrix; we
 * feed it A^T A which is at least PSD.
 *
 * Extension point: replace with LSQR or LSMR for better numerical stability.
 *
 * @tparam Scalar  float or double.
 */

#include "ILeastSquaresSolver.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class IterativeSolver : public ILeastSquaresSolver<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @param max_iter  Maximum CG iterations (0 = Eigen default).
     * @param tol       Convergence tolerance.
     */
    explicit IterativeSolver(int max_iter = 0,
                             Scalar tol = Scalar{1e-6})
        : max_iter_(max_iter), tol_(tol) {}

    Vector solve(const Matrix& A, const Vector& b) override {
        if (A.rows() < A.cols())
            throw std::invalid_argument(
                "IterativeSolver: system must be overdetermined (m >= n)");
        Matrix AtA = A.transpose() * A;
        Vector Atb = A.transpose() * b;

        Eigen::ConjugateGradient<Matrix,
                                  Eigen::Lower | Eigen::Upper> cg;
        cg.compute(AtA);
        if (max_iter_ > 0) cg.setMaxIterations(max_iter_);
        cg.setTolerance(tol_);

        Vector x = cg.solve(Atb);
        if (cg.info() != Eigen::Success) {
            // Non-fatal: return best iterate
        }
        last_iterations_ = static_cast<int>(cg.iterations());
        last_error_      = static_cast<Scalar>(cg.error());
        return x;
    }

    /// Number of CG iterations in the last solve.
    int last_iterations() const { return last_iterations_; }

    /// Estimated error in the last solve.
    Scalar last_error() const { return last_error_; }

private:
    int    max_iter_;
    Scalar tol_;
    mutable int    last_iterations_ = 0;
    mutable Scalar last_error_      = Scalar{0};
};

} // namespace randnla