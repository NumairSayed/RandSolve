#pragma once
/**
 * @file StdRNG.hpp
 * @brief Standard-library RNG implementation using <random>.
 *
 * Backed by std::mt19937_64 (64-bit Mersenne Twister).
 * Thread-unsafe by design — create one instance per thread.
 *
 * @tparam Scalar  float or double.
 */

#include "IRNG.hpp"
#include <random>
#include <cstdint>

namespace randnla {

template <typename Scalar>
class StdRNG : public IRNG<Scalar> {
public:
    /**
     * @param seed  Initial seed.  Default 42 for reproducibility.
     */
    explicit StdRNG(uint64_t seed = 42)
        : engine_(seed),
          normal_dist_(Scalar{0}, Scalar{1}),
          uniform_dist_(Scalar{0}, Scalar{1}) {}

    Scalar normal() override {
        return normal_dist_(engine_);
    }

    Scalar uniform() override {
        return uniform_dist_(engine_);
    }

    int rademacher() override {
        // Bernoulli(0.5) → {0,1} → {-1,+1}
        return (bernoulli_dist_(engine_)) ? 1 : -1;
    }

    int64_t integer(int64_t upper_exclusive) override {
        std::uniform_int_distribution<int64_t> d(0, upper_exclusive - 1);
        return d(engine_);
    }

    void seed(uint64_t s) override {
        engine_.seed(s);
        // Reset distributions (they carry no state for mt19937)
    }

private:
    std::mt19937_64                         engine_;
    std::normal_distribution<Scalar>        normal_dist_;
    std::uniform_real_distribution<Scalar>  uniform_dist_;
    std::bernoulli_distribution             bernoulli_dist_{0.5};
};

} // namespace randnla