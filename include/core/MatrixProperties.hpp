#pragma once
/**
 * @file MatrixProperties.hpp
 * @brief Lightweight structure describing the properties of a matrix,
 *        used by the DecisionEngine to select the best algorithm.
 */

#include <cstdint>
#include <limits>

namespace randnla {

/**
 * @brief Compact descriptor of a matrix's numerical and structural properties.
 *
 * All fields have sensible defaults so callers only need to fill what they know.
 */
struct MatrixProperties {
    /// Number of rows.
    int64_t m = 0;

    /// Number of columns.
    int64_t n = 0;

    /// Fraction of non-zeros in [0, 1].  1.0 = dense.
    double density = 1.0;

    /**
     * @brief Estimated numerical rank (0 = unknown → assume full rank).
     *
     * Set to a positive integer when the caller has prior knowledge of
     * low-rank structure (e.g., from domain expertise or a cheap probe).
     */
    int64_t estimated_rank = 0;

    /**
     * @brief Estimated 2-norm condition number (0 = unknown).
     *
     * A rough estimate is sufficient for routing decisions.
     * Values > 1e12 (double) suggest ill-conditioning.
     */
    double condition_number = 0.0;

    // ----------------------------------------------------------------
    // Derived helpers
    // ----------------------------------------------------------------

    /// Aspect ratio m/n.  Large values suggest overdetermined system.
    double aspect_ratio() const {
        return (n > 0) ? static_cast<double>(m) / static_cast<double>(n) : 0.0;
    }

    /// True when matrix is heavily overdetermined (m >> n).
    bool is_tall() const { return aspect_ratio() > 5.0; }

    /// True when rank is known and significantly less than min(m,n).
    bool is_low_rank() const {
        int64_t r = (estimated_rank > 0) ? estimated_rank : 0;
        int64_t mn = std::min(m, n);
        return (r > 0) && (r < mn / 2);
    }

    /// True when matrix is sparse.
    bool is_sparse() const { return density < 0.1; }

    /// True when condition number suggests potential ill-conditioning.
    bool is_ill_conditioned() const {
        return (condition_number > 0.0) && (condition_number > 1e10);
    }
};

} // namespace randnla