#pragma once

#include "ISketch.hpp"
#include "rng/IRNG.hpp"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace randnla {

/**
 * @brief Subsampled Randomized Hadamard Transform (SRHT).
 * * Applies S = (1/√k) * P * H * D.
 * The Walsh-Hadamard Transform (H) is implemented via an in-place iterative 
 * butterfly algorithm (FWHT), achieving O(m log m) complexity.
 */
template <typename Scalar>
class SRHTSketch : public ISketch<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    SRHTSketch(int k, int m, IRNG<Scalar>& rng)
        : k_(k), m_(m), rng_(rng) {
        if ((m & (m - 1)) != 0 || m <= 0)
            throw std::invalid_argument(
                "SRHTSketch: m must be a power of 2 (e.g., 512, 1024, 2048)");
        if (k > m)
            throw std::invalid_argument("SRHTSketch: k must be <= m");

        build_diagonal();
        build_sampler();
    }

    Matrix apply(const Matrix& A) override {
        if (A.rows() != m_)
            throw std::invalid_argument("SRHTSketch::apply(Matrix): row mismatch");

        // Step 1: D * A (Randomize signs to spread signal energy)
        Matrix result = D_.asDiagonal() * A;

        // Step 2: H * (DA) via Fast Walsh-Hadamard Transform
        hadamard_transform_inplace(result);

        // Step 3: Subsample k rows and scale by 1/sqrt(k)
        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k_));
        Matrix sampled(k_, A.cols());
        for (int i = 0; i < k_; ++i) {
            sampled.row(i) = result.row(sample_rows_[i]) * scale;
        }

        return sampled;
    }

    Vector apply(const Vector& b) override {
        if (b.size() != m_)
            throw std::invalid_argument("SRHTSketch::apply(Vector): size mismatch");

        Vector result = D_.cwiseProduct(b);
        hadamard_transform_inplace_vec(result);

        const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k_));
        Vector sampled(k_);
        for (int i = 0; i < k_; ++i) {
            sampled(i) = result(sample_rows_[i]) * scale;
        }

        return sampled;
    }

    int sketch_dim() const override { return k_; }
    int input_dim()  const override { return m_; }

private:
    void build_diagonal() {
        D_.resize(m_);
        for (int i = 0; i < m_; ++i) {
            D_(i) = static_cast<Scalar>(rng_.rademacher());
        }
    }

    void build_sampler() {
        // SRHT requires sampling WITHOUT replacement to be a valid embedding.
        std::vector<int> all_indices(m_);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        // Partial Fisher-Yates shuffle to pick k unique indices
        for (int i = 0; i < k_; ++i) {
            int j = i + static_cast<int>(rng_.integer(m_ - i));
            std::swap(all_indices[i], all_indices[j]);
        }

        sample_rows_.assign(all_indices.begin(), all_indices.begin() + k_);
    }

    /**
     * @brief O(m log m) In-place Walsh-Hadamard Transform for Matrices.
     * Uses Eigen's block operations to allow internal vectorization across columns.
     */
    void hadamard_transform_inplace(Matrix& M) {
        const int m = static_cast<int>(M.rows());
        for (int h = 1; h < m; h <<= 1) {
            for (int i = 0; i < m; i += (h << 1)) {
                // Process blocks of rows: [top; bottom] -> [top + bottom; top - bottom]
                auto top = M.middleRows(i, h);
                auto bot = M.middleRows(i + h, h);
                
                // We need a temporary copy of 'top' to perform the subtraction
                Matrix temp = top;
                top += bot;
                bot = temp - bot;
            }
        }
    }

    /**
     * @brief O(m log m) In-place Walsh-Hadamard Transform for Vectors.
     */
    void hadamard_transform_inplace_vec(Vector& v) {
        const int m = static_cast<int>(v.size());
        for (int h = 1; h < m; h <<= 1) {
            for (int i = 0; i < m; i += (h << 1)) {
                for (int j = i; j < i + h; ++j) {
                    Scalar x = v(j);
                    Scalar y = v(j + h);
                    v(j)     = x + y;
                    v(j + h) = x - y;
                }
            }
        }
    }

    int               k_, m_;
    IRNG<Scalar>&     rng_;
    Vector            D_;            ///< Rademacher diagonal
    std::vector<int>  sample_rows_;  ///< K distinct sampled indices
};

} // namespace randnla