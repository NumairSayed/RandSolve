#pragma once

#include "MatrixProperties.hpp"
#include <cstdint>

namespace randnla {

enum class AlgorithmChoice {
    QR_DIRECT,
    NORMAL_EQUATIONS,
    SKETCHED_LS,
    PRECONDITIONED_LS,
    ITERATIVE_CG,
    RANDOMIZED_SVD,
};

inline const char* to_string(AlgorithmChoice c) {
    switch (c) {
        case AlgorithmChoice::QR_DIRECT:         return "QR_DIRECT";
        case AlgorithmChoice::NORMAL_EQUATIONS:  return "NORMAL_EQUATIONS";
        case AlgorithmChoice::SKETCHED_LS:       return "SKETCHED_LS";
        case AlgorithmChoice::PRECONDITIONED_LS: return "PRECONDITIONED_LS";
        case AlgorithmChoice::ITERATIVE_CG:      return "ITERATIVE_CG";
        case AlgorithmChoice::RANDOMIZED_SVD:    return "RANDOMIZED_SVD";
        default:                                 return "UNKNOWN";
    }
}

// Config lives at namespace scope — avoids default-argument/in-class-initializer
// ordering issue on older GCC versions.
struct DecisionEngineConfig {
    double  tall_threshold         = 10.0;
    int64_t large_sparse_threshold = 100000;
    int64_t small_direct_threshold = 2000;
};

class DecisionEngine {
public:
    // Keep nested alias for backwards compatibility with LeastSquaresDriver
    using Config = DecisionEngineConfig;

    explicit DecisionEngine(DecisionEngineConfig cfg = DecisionEngineConfig())
        : cfg_(cfg) {}

    AlgorithmChoice route(const MatrixProperties& props) const {
        if (props.is_low_rank())
            return AlgorithmChoice::RANDOMIZED_SVD;

        int64_t mn = std::min(props.m, props.n);
        if (props.is_sparse() && mn > cfg_.large_sparse_threshold)
            return AlgorithmChoice::ITERATIVE_CG;

        if (props.is_tall() && !props.is_ill_conditioned())
            if (props.aspect_ratio() > cfg_.tall_threshold)
                return AlgorithmChoice::SKETCHED_LS;

        if (props.is_ill_conditioned())
            return AlgorithmChoice::PRECONDITIONED_LS;

        if (props.is_tall() && !props.is_sparse())
            return AlgorithmChoice::NORMAL_EQUATIONS;

        return AlgorithmChoice::QR_DIRECT;
    }

    const DecisionEngineConfig& config() const { return cfg_; }

private:
    DecisionEngineConfig cfg_;
};

} // namespace randnla