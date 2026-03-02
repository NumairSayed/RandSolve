#pragma once
/**
 * @file IRNG.hpp
 * @brief Abstract random number generator interface.
 *
 * Sketchers depend ONLY on this interface, never on a concrete RNG.
 * This enables:
 *   - Seeded reproducibility (StdRNG with fixed seed)
 *   - Fast Eigen-based generation (EigenRNG)
 *   - Mock RNGs in unit tests
 *
 * @tparam Scalar  Floating-point type for continuous variates.
 */

#include <cstdint>

namespace randnla {

template <typename Scalar>
class IRNG {
public:
    virtual ~IRNG() = default;

    /// Draw a standard normal variate N(0,1).
    virtual Scalar normal() = 0;

    /// Draw a uniform variate U(0,1).
    virtual Scalar uniform() = 0;

    /**
     * @brief Draw a Rademacher variate: returns +1 or -1 with equal probability.
     *
     * Used in sparse embedding and SRHT constructions.
     */
    virtual int rademacher() = 0;

    /**
     * @brief Draw a random integer in [0, upper_exclusive).
     *
     * Default implementation uses uniform(); subclasses may override.
     */
    virtual int64_t integer(int64_t upper_exclusive) {
        return static_cast<int64_t>(uniform() * static_cast<Scalar>(upper_exclusive))
               % upper_exclusive;
    }

    /// Reseed the generator for reproducibility.
    virtual void seed(uint64_t s) = 0;
};

} // namespace randnla