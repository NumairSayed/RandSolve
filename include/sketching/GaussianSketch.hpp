#pragma once
/**
 * @file GaussianSketch.hpp
 * @brief Gaussian (Johnson-Lindenstrauss) sketching operator.
 *
 * Constructs S ∈ R^{k×m} with i.i.d. entries S_{ij} ~ N(0, 1/k).
 * The 1/k scaling ensures E[||SAx||^2] = ||Ax||^2.
 *
 * Embedding guarantee: For any fixed x ∈ R^n, with probability ≥ 1-δ,
 *   (1-ε)||Ax|| ≤ ||SAx|| ≤ (1+ε)||Ax||
 * when k = O(ε^{-2} log(1/δ)).
 *
 * Reference: Johnson & Lindenstrauss (1984).
 *
 * @tparam Scalar  float or double.
 */

#include "ISketch.hpp"
#include "rng/IRNG.hpp"
#include <cmath>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class GaussianSketch : public ISketch<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @param k    Target sketch dimension (rows of S).
     * @param m    Input dimension (columns of S = rows of A).
     * @param rng  RNG instance; not owned — caller manages lifetime.
     */
    GaussianSketch(int k, int m, IRNG<Scalar>& rng)
        : k_(k), m_(m), rng_(rng) {
        if (k <= 0 || m <= 0)
            throw std::invalid_argument("GaussianSketch: k and m must be positive");
        if (k > m)
            throw std::invalid_argument("GaussianSketch: k must be <= m");
        build_sketch_matrix();
    }

    Matrix apply(const Matrix& A) override {
        if (A.rows() != m_)
            throw std::invalid_argument("GaussianSketch::apply: row mismatch");
        return S_ * A;
    }

    Vector apply(const Vector& b) override {
        if (b.size() != m_)
            throw std::invalid_argument("GaussianSketch::apply: size mismatch");
        return S_ * b;
    }

    int sketch_dim() const override { return k_; }
    int input_dim()  const override { return m_; }

    /// Return a const reference to the explicit sketch matrix S.
    const Matrix& sketch_matrix() const { return S_; }

    /// Regenerate S with fresh random entries (e.g., for bootstrap resampling).
    void resample() { build_sketch_matrix(); }

private:
    void build_sketch_matrix() {
        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k_));
        S_.resize(k_, m_);
        for (int i = 0; i < k_; ++i)
            for (int j = 0; j < m_; ++j)
                S_(i, j) = scale * rng_.normal();
    }

    int          k_;
    int          m_;
    IRNG<Scalar>& rng_;
    Matrix        S_;
};

} // namespace randnla