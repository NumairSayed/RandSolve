#pragma once
/**
 * @file RandomizedSVD.hpp
 * @brief Randomized SVD via range-finder + deterministic SVD.
 *
 * Algorithm (Halko, Martinsson, Tropp 2011):
 *   1. Draw Ω ∈ R^{n×(k+p)}  (Gaussian random matrix, p = oversampling)
 *   2. Form Y = A Ω
 *   3. (Optional) Power iteration: Y = (A A^T)^q  Y  for improved accuracy
 *   4. Orthogonalize Y → Q  via QR
 *   5. Form B = Q^T A  (small (k+p) × n matrix)
 *   6. SVD of B: B = Û Σ V^T
 *   7. Recover U = Q Û
 *
 * Reference:
 *   Halko, Martinsson, Tropp (2011) "Finding Structure with Randomness"
 *   SIAM Review 53(2), pp. 217-288.
 *
 * @tparam Scalar  float or double.
 */

#include "ILowRankApproximation.hpp"
#include "rng/IRNG.hpp"
#include <Eigen/SVD>
#include <Eigen/QR>
#include <stdexcept>
#include <cmath>

namespace randnla {

template <typename Scalar>
class RandomizedSVD : public ILowRankApproximation<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @param oversampling  p: extra columns in sketch (default 10).
     * @param power_iters   q: power iteration count (0 = no power iter).
     */
    explicit RandomizedSVD(int oversampling = 10, int power_iters = 2)
        : oversampling_(oversampling), power_iters_(power_iters) {}

    /**
     * @brief Compute rank-k approximation using provided RNG.
     */
    void compute(const Matrix& A, int rank, IRNG<Scalar>& rng) {
        int m = static_cast<int>(A.rows());
        int n = static_cast<int>(A.cols());
        int l = rank + oversampling_;
        l = std::min(l, std::min(m, n));

        // Step 1: Random Gaussian sketch Ω ∈ R^{n×l}
        Matrix Omega(n, l);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < l; ++j)
                Omega(i, j) = rng.normal();

        // Step 2: Y = A * Ω
        Matrix Y = A * Omega;

        // Step 3: Power iteration for better spectral decay
        for (int iter = 0; iter < power_iters_; ++iter) {
            // Re-orthogonalize to prevent numerical drift
            Y = A * (A.transpose() * Y);
        }

        // Step 4: QR decomposition of Y → Q ∈ R^{m×l}
        Eigen::HouseholderQR<Matrix> qr(Y);
        Matrix Q = qr.householderQ() * Matrix::Identity(m, l);

        // Step 5: B = Q^T * A  ∈ R^{l×n}
        Matrix B = Q.transpose() * A;

        // Step 6: Thin SVD of B
        Eigen::JacobiSVD<Matrix> svd(B,
            Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Step 7: Recover full left singular vectors
        int k = std::min(rank, static_cast<int>(svd.singularValues().size()));
        U_     = Q * svd.matrixU().leftCols(k);
        sigma_ = svd.singularValues().head(k);
        Vt_    = svd.matrixV().leftCols(k).transpose();
        computed_ = true;
    }

    // ILowRankApproximation interface
    void compute(const Matrix& A, int rank) override {
    (void)A; (void)rank;
    throw std::runtime_error(
        "RandomizedSVD::compute(A, rank): must provide an IRNG. "
        "Call compute(A, rank, rng) instead.");
    }

    Matrix matrixU()       const override { check(); return U_;  }
    Vector singularValues()const override { check(); return sigma_; }
    Matrix matrixVt()      const override { check(); return Vt_; }

    /// Reconstruct the approximation A_k ≈ U Σ V^T.
    Matrix reconstruct() const {
        check();
        return U_ * sigma_.asDiagonal() * Vt_;
    }

private:
    void check() const {
        if (!computed_)
            throw std::runtime_error("RandomizedSVD: call compute() first");
    }

    int    oversampling_;
    int    power_iters_;
    bool   computed_ = false;
    Matrix U_;
    Vector sigma_;
    Matrix Vt_;
};

} // namespace randnla