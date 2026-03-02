#pragma once
/**
 * @file LowRankDriver.hpp
 * @brief High-level driver for low-rank approximation tasks.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "lowrank/RandomizedSVD.hpp"
#include "rng/StdRNG.hpp"
#include <memory>

namespace randnla {

template <typename Scalar>
class LowRankDriver {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    struct Result {
        Matrix U;     ///< Left singular vectors,  shape (m, k)
        Vector sigma; ///< Singular values,         length k
        Matrix Vt;    ///< Right singular vectors^T, shape (k, n)
    };

    struct Options {
        int      oversampling  = 10;   ///< Extra sketch columns beyond k
        int      power_iters   = 2;    ///< Power iteration count for quality
        uint64_t rng_seed      = 42;
    };

    explicit LowRankDriver(Options opts = {}) : opts_(opts) {}

    Result compute(const Matrix& A, int rank) {
        rng_ = std::make_unique<StdRNG<Scalar>>(opts_.rng_seed);
        RandomizedSVD<Scalar> rsvd(opts_.oversampling, opts_.power_iters);
        rsvd.compute(A, rank, *rng_);
        return Result{rsvd.matrixU(), rsvd.singularValues(), rsvd.matrixVt()};
    }

private:
    Options                         opts_;
    std::unique_ptr<StdRNG<Scalar>> rng_;
};

} // namespace randnla