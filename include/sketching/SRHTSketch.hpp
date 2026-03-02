#pragma once
/**
 * @file SRHTSketch.hpp
 * @brief Subsampled Randomized Hadamard Transform (SRHT) sketch — placeholder.
 *
 * The SRHT applies:
 *   S = (1/√k) · P · H · D
 * where:
 *   D ∈ R^{m×m}  diagonal Rademacher matrix
 *   H ∈ R^{m×m}  Walsh-Hadamard transform  (requires m = power of 2)
 *   P ∈ R^{k×m}  uniform random row sampler
 *
 * Time complexity: O(m·n·log(k))  vs  O(k·m·n) for Gaussian.
 *
 * Embedding quality matches Gaussian sketch asymptotically.
 * Reference: Ailon & Chazelle (2006), Tropp et al. (2011).
 *
 * STATUS: Scaffold — Hadamard transform body is a TODO.
 *         The interface is complete and the class compiles.
 *
 * @tparam Scalar  float or double.
 */

#include "ISketch.hpp"
#include "rng/IRNG.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class SRHTSketch : public ISketch<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    SRHTSketch(int k, int m, IRNG<Scalar>& rng)
        : k_(k), m_(m), rng_(rng) {
        if ((m & (m - 1)) != 0)
            throw std::invalid_argument(
                "SRHTSketch: m must be a power of 2 for Walsh-Hadamard transform");
        if (k > m)
            throw std::invalid_argument("SRHTSketch: k must be <= m");
        build_diagonal();
        build_sampler();
    }

    Matrix apply(const Matrix& A) override {
        if (A.rows() != m_)
            throw std::invalid_argument("SRHTSketch::apply(Matrix): row mismatch");
        // Step 1: D*A
        Matrix DA = D_.asDiagonal() * A;
        // Step 2: H*DA  (TODO: replace with fast WHT)
        hadamard_transform_inplace(DA);
        // Step 3: Subsample rows
        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k_));
        Matrix result(k_, A.cols());
        for (int i = 0; i < k_; ++i)
            result.row(i) = scale * DA.row(sample_rows_[i]);
        return result;
    }

    Vector apply(const Vector& b) override {
        if (b.size() != m_)
            throw std::invalid_argument("SRHTSketch::apply(Vector): size mismatch");
        Vector Db = D_.cwiseProduct(b);
        hadamard_transform_inplace_vec(Db);
        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k_));
        Vector result(k_);
        for (int i = 0; i < k_; ++i)
            result(i) = scale * Db(sample_rows_[i]);
        return result;
    }

    int sketch_dim() const override { return k_; }
    int input_dim()  const override { return m_; }

private:
    void build_diagonal() {
        D_.resize(m_);
        for (int i = 0; i < m_; ++i)
            D_(i) = static_cast<Scalar>(rng_.rademacher());
    }

    void build_sampler() {
        // Sample k distinct rows uniformly at random (with replacement for simplicity)
        sample_rows_.resize(k_);
        for (int i = 0; i < k_; ++i)
            sample_rows_[i] = static_cast<int>(rng_.integer(m_));
    }

    /**
     * @brief In-place Walsh-Hadamard transform (iterative Cooley-Tukey style).
     * TODO: Implement full O(m log m) WHT.
     */
    void hadamard_transform_inplace(Matrix& M) {
        // Naive O(m^2) placeholder — replace with fast WHT for production.
        int m = static_cast<int>(M.rows());
        for (int step = 1; step < m; step <<= 1) {
            for (int i = 0; i < m; i += step << 1) {
                for (int j = i; j < i + step; ++j) {
                    auto u = M.row(j);
                    auto v = M.row(j + step);
                    M.row(j)        = u + v;
                    M.row(j + step) = u - v;
                }
            }
        }
    }

    void hadamard_transform_inplace_vec(Vector& v) {
        int m = static_cast<int>(v.size());
        for (int step = 1; step < m; step <<= 1) {
            for (int i = 0; i < m; i += step << 1) {
                for (int j = i; j < i + step; ++j) {
                    Scalar u = v(j), w = v(j + step);
                    v(j)        = u + w;
                    v(j + step) = u - w;
                }
            }
        }
    }

    int               k_, m_;
    IRNG<Scalar>&     rng_;
    Vector            D_;            ///< Rademacher diagonal
    std::vector<int>  sample_rows_;  ///< Sampled row indices
};

} // namespace randnla