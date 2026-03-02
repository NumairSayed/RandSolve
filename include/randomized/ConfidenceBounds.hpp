#pragma once
/**
 * @file ConfidenceBounds.hpp
 * @brief Error and confidence bound estimation for randomized least squares.
 *
 * Provides two complementary approaches:
 *
 * A) Analytic (subspace embedding) bound:
 *    When S is a (1+ε)-subspace embedding for col(A):
 *      ||x̃ - x*|| ≤ 2ε/(1-ε) · ||x*||   (under mild conditions)
 *    This is a worst-case bound.  ε depends on k and the sketch type.
 *
 * B) Empirical (bootstrap-style) estimate:
 *    Re-sketch multiple times, collect solutions {x̃_i}, and estimate
 *    std-dev as a proxy for solution variability.
 *    NOTE: This is a scaffold — computation is stubbed.
 *
 * Reference:
 *   Woodruff (2014) "Sketching as a Tool for NLA", Section 2.3.
 *   Drineas, Mahoney et al. (2011) "Faster Least Squares Approximation".
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "SketchedLeastSquares.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace randnla {

/// Result structure for bound estimation.
template <typename Scalar>
struct BoundResult {
    Scalar residual_norm;       ///< ||Ax̃ - b||_2
    Scalar relative_error;      ///< ||Ax̃ - b||_2 / ||b||_2
    Scalar probabilistic_bound; ///< Analytic ε-embedding bound (pessimistic)

    /// Empirical std-dev across bootstrap resamples (0 if not computed).
    Scalar bootstrap_std = Scalar{0};

    /// Number of bootstrap trials used (0 if analytic only).
    int bootstrap_trials = 0;
};

template <typename Scalar>
class ConfidenceBounds {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @brief Compute analytic bound for a given solution x̃.
     *
     * @param A            Original system matrix.
     * @param b            Right-hand side.
     * @param x_approx     Approximate solution from sketch-and-solve.
     * @param epsilon      Embedding distortion parameter ε (see sketch docs).
     */
    static BoundResult<Scalar> analytic_bound(const Matrix& A,
                                               const Vector& b,
                                               const Vector& x_approx,
                                               Scalar epsilon) {
        BoundResult<Scalar> result;
        result.residual_norm = (A * x_approx - b).norm();
        Scalar b_norm = b.norm();
        result.relative_error = (b_norm > Scalar{0})
                                    ? result.residual_norm / b_norm
                                    : Scalar{0};
        // Pessimistic bound: (1+ε)·opt_residual ≤ approx_residual ≤ (1+ε)·opt_residual
        // Here we report the multiplicative factor as the bound.
        result.probabilistic_bound = (Scalar{1} + epsilon) * result.residual_norm;
        return result;
    }

    /**
     * @brief Bootstrap-style empirical bound.
     *
     * Re-runs sketched LS `trials` times with independent sketch redraws
     * (via SketchedLeastSquares::solve), collects solutions, and estimates
     * std-dev across trials.
     *
     * @param sls       SketchedLeastSquares instance (will be called repeatedly).
     * @param A         System matrix.
     * @param b         Right-hand side.
     * @param trials    Number of bootstrap redraws.
     *
     * STATUS: Scaffold — the SLS object must expose a re-sketch mechanism.
     *         Currently calls solve() repeatedly (each call redraws the sketch).
     */
    static BoundResult<Scalar> bootstrap_bound(SketchedLeastSquares<Scalar>& sls,
                                                const Matrix& A,
                                                const Vector& b,
                                                int trials = 20) {
        if (trials < 2)
            throw std::invalid_argument("bootstrap_bound: need at least 2 trials");

        std::vector<Vector> solutions;
        solutions.reserve(trials);
        for (int t = 0; t < trials; ++t) {
            solutions.push_back(sls.solve(A, b));
        }

        // Compute mean solution
        Vector mean = Vector::Zero(solutions[0].size());
        for (const auto& x : solutions) mean += x;
        mean /= static_cast<Scalar>(trials);

        // Compute std-dev across trials (element-wise mean of ||x_i - mean||)
        Scalar var = Scalar{0};
        for (const auto& x : solutions)
            var += (x - mean).squaredNorm();
        var /= static_cast<Scalar>(trials - 1);

        BoundResult<Scalar> result;
        // Use mean solution for residual
        result.residual_norm  = (A * mean - b).norm();
        Scalar b_norm = b.norm();
        result.relative_error = (b_norm > Scalar{0})
                                    ? result.residual_norm / b_norm
                                    : Scalar{0};
        result.probabilistic_bound = result.residual_norm; // no analytic ε here
        result.bootstrap_std       = std::sqrt(var);
        result.bootstrap_trials    = trials;
        return result;
    }
};

} // namespace randnla