#pragma once
/**
 * @file SparseSketch.hpp
 * @brief Sparse CountSketch / OSNAP embedding.
 *
 * Each column of S has exactly `s` non-zeros drawn uniformly from {+1/√s, -1/√s}.
 * When s=1 this is the CountSketch of Clarkson & Woodruff (2013).
 *
 * Time complexity: O(s·m·n) vs O(k·m·n) for Gaussian sketch.
 * Embedding dimension: k = O(n^2 / ε^2) for s=1; improves with s.
 *
 * @tparam Scalar  float or double.
 */

#include "ISketch.hpp"
#include "rng/IRNG.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class SparseSketch : public ISketch<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    /**
     * @param k  Target sketch rows.
     * @param m  Input rows (columns of S).
     * @param s  Non-zeros per column of S (s=1 → CountSketch).
     * @param rng  RNG reference.
     */
    SparseSketch(int k, int m, IRNG<Scalar>& rng, int s = 1)
        : k_(k), m_(m), s_(s), rng_(rng) {
        if (k <= 0 || m <= 0 || s <= 0)
            throw std::invalid_argument("SparseSketch: invalid dimensions");
        if (s > k)
            throw std::invalid_argument("SparseSketch: s must be <= k");
        build();
    }

    Matrix apply(const Matrix& A) override {
        if (A.rows() != m_)
            throw std::invalid_argument("SparseSketch::apply(Matrix): row mismatch");
        Matrix result = Matrix::Zero(k_, A.cols());
        for (int j = 0; j < m_; ++j) {
            for (int t = 0; t < s_; ++t) {
                int    row  = rows_[j * s_ + t];
                Scalar sign = signs_[j * s_ + t];
                result.row(row) += sign * A.row(j);
            }
        }
        return result;
    }

    Vector apply(const Vector& b) override {
        if (b.size() != m_)
            throw std::invalid_argument("SparseSketch::apply(Vector): size mismatch");
        Vector result = Vector::Zero(k_);
        for (int j = 0; j < m_; ++j) {
            for (int t = 0; t < s_; ++t) {
                int    row  = rows_[j * s_ + t];
                Scalar sign = signs_[j * s_ + t];
                result(row) += sign * b(j);
            }
        }
        return result;
    }

    int sketch_dim() const override { return k_; }
    int input_dim()  const override { return m_; }

private:
    void build() {
        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(s_));
        rows_.resize(m_ * s_);
        signs_.resize(m_ * s_);
        for (int j = 0; j < m_; ++j) {
            for (int t = 0; t < s_; ++t) {
                rows_[j * s_ + t]  = static_cast<int>(rng_.integer(k_));
                signs_[j * s_ + t] = static_cast<Scalar>(rng_.rademacher()) * scale;
            }
        }
    }

    int               k_, m_, s_;
    IRNG<Scalar>&     rng_;
    std::vector<int>  rows_;
    std::vector<Scalar> signs_;
};

} // namespace randnla