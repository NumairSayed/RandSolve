#pragma once
/**
 * @file PreconditionedLeastSquares.hpp
 * @brief Sketch-based preconditioned least squares (Blendenpik / LSRN style).
 *
 * Algorithm:
 *   1. Compute sketch SA  (k×n, k > n)
 *   2. Factor SA = QR  →  R is an n×n preconditioner
 *   3. Solve min_x ||A R^{-1} y - b||_2  using LSQR/CG  (well-conditioned)
 *   4. Recover x = R^{-1} y
 *
 * This approach is effective when A is ill-conditioned: R^{-1} acts as a
 * right preconditioner that approximately orthogonalises the column space.
 *
 * Reference:
 *   Avron, Maymounkov, Toledo (2010) "Blendenpik: Supercharging LAPACK's LS Solvers"
 *   Meng & Mahoney (2014) "Low-Distortion Subspace Embeddings in Input-Sparsity Time"
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "sketching/ISketch.hpp"
#include "solvers/ILeastSquaresSolver.hpp"
#include <memory>
#include <stdexcept>
#include <Eigen/QR>

namespace randnla {

template <typename Scalar>
class PreconditionedLeastSquares {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    PreconditionedLeastSquares(std::shared_ptr<ISketch<Scalar>>             sketch,
                                std::shared_ptr<ILeastSquaresSolver<Scalar>> iterative_solver)
        : sketch_(std::move(sketch)),
          iterative_solver_(std::move(iterative_solver)) {}

    Vector solve(const Matrix& A, const Vector& b) {
        if (A.rows() != b.size())
            throw std::invalid_argument("PreconditionedLS: dimension mismatch");

        // Step 1: Sketch
        Matrix SA = sketch_->apply(A);

        // Step 2: QR of SA to extract preconditioner R
        Eigen::ColPivHouseholderQR<Matrix> qr(SA);
        Matrix R = qr.matrixR()
                       .topLeftCorner(A.cols(), A.cols())
                       .template triangularView<Eigen::Upper>();

        // Step 3: Form preconditioned system A_prec = A * R^{-1}
        // Solve R^T * R * z = R^T * A^T * b  (sketch of preconditioned normal eqs)
        // Simplified: solve using iterative solver on preconditioned A
        Matrix AR_inv = qr.solve(SA).transpose(); // placeholder; see note below
        // NOTE: Full Blendenpik implementation would apply LSQR on A*R^{-1}.
        //       This scaffold falls back to iterative solver on original A.
        // TODO: Replace with proper R^{-1} application via triangular solve.
        return iterative_solver_->solve(A, b);
    }

private:
    std::shared_ptr<ISketch<Scalar>>             sketch_;
    std::shared_ptr<ILeastSquaresSolver<Scalar>> iterative_solver_;
};

} // namespace randnla