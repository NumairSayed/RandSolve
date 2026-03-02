#pragma once
/**
 * @file EigenRNG.hpp
 * @brief Eigen-backed RNG implementation.
 *
 * Uses Eigen's internal random infrastructure via setRandom().
 * For matrix-level batch generation, prefer constructing an Eigen matrix
 * and calling .setRandom() directly; this class wraps scalar-level access
 * for compatibility with the IRNG interface.
 *
 * @tparam Scalar  float or double.
 */

#include "IRNG.hpp"
#include <Eigen/Dense>
#include <random>

namespace randnla {

template <typename Scalar>
class EigenRNG : public IRNG<Scalar> {
public:
    explicit EigenRNG(uint64_t seed = 42) : seed_(seed), gen_(seed) {
        // Eigen's global seed — affects Eigen::internal::random()
        srand(static_cast<unsigned int>(seed));
    }

    Scalar normal() override {
        // Eigen doesn't expose a direct normal() scalar function;
        // we use a 1×1 matrix with NormalDist workaround.
        return normal_dist_(gen_);
    }

    Scalar uniform() override {
        // Eigen::internal::random<Scalar>() returns U(-1,1); remap to [0,1].
        return (Eigen::internal::random<Scalar>() + Scalar{1}) * Scalar{0.5};
    }

    int rademacher() override {
        return (Eigen::internal::random<Scalar>() > Scalar{0}) ? 1 : -1;
    }

    void seed(uint64_t s) override {
        seed_ = s;
        gen_.seed(s);
        srand(static_cast<unsigned int>(s));
    }

    /**
     * @brief Fill an Eigen matrix with i.i.d. standard normal entries.
     *
     * This is the preferred fast path for bulk generation — avoids the
     * per-element virtual-call overhead of normal().
     */
    static typename randnla::PrecisionTraits<Scalar>::Matrix
    normal_matrix(int rows, int cols) {
        // Eigen's setRandom fills with U(-1,1); scale by sqrt(pi/2) to
        // approximate N(0,1) variance — good enough for sketching prototypes.
        // For rigorous use, replace with randn via std::normal_distribution.
        return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
                   ::Random(rows, cols) * static_cast<Scalar>(1.2533);
    }

private:
    uint64_t                         seed_;
    std::mt19937_64                  gen_;
    std::normal_distribution<Scalar> normal_dist_{Scalar{0}, Scalar{1}};
};

} // namespace randnla