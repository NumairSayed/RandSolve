#pragma once
/**
 * @file SketchedLeastSquares.hpp
 * @brief Sketch-and-solve least squares.
 *
 * Algorithm:
 *   Given A ∈ R^{m×n},  b ∈ R^m,  sketch dimension k < m:
 *
 *   1.  Draw S ∈ R^{k×m}  (via ISketch)
 *   2.  Compute SA ∈ R^{k×n},  Sb ∈ R^k
 *   3.  Solve  min_x ||SA·x - Sb||_2  (via ILeastSquaresSolver)
 *   4.  Return x̃  (approximate solution to original problem)
 *
 * Quality guarantee: When S is a (ε, δ)-subspace embedding for
 * col(A), the solution x̃ satisfies
 *     ||Ax̃ - b|| ≤ (1+ε) · ||Ax* - b||
 * with probability ≥ 1 - δ.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "sketching/ISketch.hpp"
#include "solvers/ILeastSquaresSolver.hpp"
#include <memory>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class SketchedLeastSquares {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @param sketch  Sketching operator (shared ownership).
     * @param solver  Deterministic solver applied to the sketched problem (shared ownership).
     */
    SketchedLeastSquares(std::shared_ptr<ISketch<Scalar>>             sketch,
                          std::shared_ptr<ILeastSquaresSolver<Scalar>> solver)
        : sketch_(std::move(sketch)), solver_(std::move(solver)) {
        if (!sketch_ || !solver_)
            throw std::invalid_argument("SketchedLeastSquares: null sketch or solver");
    }

    /**
     * @brief Solve min_x ||Ax - b||_2 via sketch-and-solve.
     */
    Vector solve(const Matrix& A, const Vector& b) {
        if (A.rows() != b.size())
            throw std::invalid_argument("SketchedLeastSquares: A.rows() != b.size()");
        if (A.rows() != sketch_->input_dim())
            throw std::invalid_argument(
                "SketchedLeastSquares: A.rows() != sketch input_dim");

        SA_ = sketch_->apply(A);
        Sb_ = sketch_->apply(b);

        return solver_->solve(SA_, Sb_);
    }

    /// Residual norm ||Ax - b||_2 using the original system.
    Scalar residual_norm(const Matrix& A, const Vector& b, const Vector& x) const {
        return (A * x - b).norm();
    }

    /// Access the last computed sketched matrix (for diagnostics).
    const Matrix& last_SA() const { return SA_; }
    const Vector& last_Sb() const { return Sb_; }

private:
    std::shared_ptr<ISketch<Scalar>>             sketch_;
    std::shared_ptr<ILeastSquaresSolver<Scalar>> solver_;
    Matrix SA_;
    Vector Sb_;
};

} // namespace randnla