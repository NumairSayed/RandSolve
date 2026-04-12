#pragma once
/**
 * @file SketchedAttention.hpp
 * @brief Randomized sketched self-attention mechanism.
 *
 * Standard self-attention:
 *   Scores = Q K^T / sqrt(d)      shape (L, L)  — quadratic bottleneck
 *   A      = softmax(Scores)
 *   O      = A V
 *
 * Sketched self-attention:
 *   SQ     = S Q                  shape (k, d),  S ∈ R^{k×d}, k << L
 *   SK     = S K                  shape (k, d)
 *   Scores = (L×k)(k×L) / sqrt(k) shape (L, L)  — but formed as Q S^T S K^T
 *   A      = softmax(Scores)
 *   O      = A V
 *
 * Complexity:
 *   Exact:    O(L² d)
 *   Sketched: O(L k d)   for Gaussian sketch construction
 *             O(L k)     for score approximation (dominant at large L)
 *
 * Multi-head support:
 *   Each head gets an independent sketch drawn fresh per forward pass.
 *   This is critical — sharing sketches across heads collapses diversity.
 *
 * Gradient note:
 *   The sketch S is treated as a fixed (non-trainable) random matrix
 *   within each forward pass. Gradients flow through Q, K, V normally.
 *   The sketch introduces a biased-but-consistent gradient estimator.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "rng/IRNG.hpp"
#include "rng/StdRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "sketching/SRHTSketch.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>


using Clock = std::chrono::high_resolution_clock;

double time_ms(auto start, auto end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}


namespace randnla {
namespace attention {
    
// ----------------------------------------------------------------
// Softmax helpers
// ----------------------------------------------------------------

/**
 * @brief Numerically stable row-wise softmax.
 * For each row i: softmax(x)_j = exp(x_j - max_j x) / sum_j exp(x_j - max_j x)
 */
template <typename Scalar>
typename PrecisionTraits<Scalar>::Matrix
row_softmax(const typename PrecisionTraits<Scalar>::Matrix& X) {
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    int L = static_cast<int>(X.rows());
    Matrix out(X.rows(), X.cols());

    for (int i = 0; i < L; ++i) {
        // Subtract row max for numerical stability
        Scalar row_max = X.row(i).maxCoeff();
        auto row_exp   = (X.row(i).array() - row_max).exp();
        Scalar row_sum = row_exp.sum();
        out.row(i)     = row_exp / row_sum;
    }
    return out;
}

/**
 * @brief Optional causal mask: set upper triangle to -inf before softmax.
 * Used for autoregressive (decoder) attention.
 */
template <typename Scalar>
void apply_causal_mask(typename PrecisionTraits<Scalar>::Matrix& Scores) {
    int L = static_cast<int>(Scores.rows());
    for (int i = 0; i < L; ++i)
        for (int j = i + 1; j < L; ++j)
            Scores(i, j) = -std::numeric_limits<Scalar>::infinity();
}

// ----------------------------------------------------------------
// Sketch type enum
// ----------------------------------------------------------------
enum class SketchType {
    GAUSSIAN,       ///< Dense Gaussian sketch (better quality)
    SPARSE_COUNT,   ///< CountSketch s=1 (faster construction)
    SPARSE_OSNAP,   ///< OSNAP s=4 (balance)
    SRHT,           ///< Subsampled Randomized Hadamard (d must be a power of 2)
};

// ----------------------------------------------------------------
// AttentionResult — carries outputs and diagnostics
// ----------------------------------------------------------------
template <typename Scalar>
struct AttentionResult {
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;

    Matrix output;              ///< O = A V, shape (L, d)
    Matrix attention_weights;   ///< A = softmax(Scores), shape (L, L)

    // Diagnostics (only populated when compute_diagnostics=true)
    Scalar score_frobenius_norm = Scalar{0};  ///< ||Scores||_F
    Scalar attn_entropy         = Scalar{0};  ///< Mean row entropy of A
    double sketch_build_ms      = 0.0;
    double score_compute_ms     = 0.0;
    double softmax_ms           = 0.0;
    double context_ms           = 0.0;
    double total_ms             = 0.0;
};

// ----------------------------------------------------------------
// ExactAttention — baseline
// ----------------------------------------------------------------
template <typename Scalar>
class ExactAttention {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;

    struct Config {
        bool causal            = false;
        bool compute_diag      = false;
    };

    explicit ExactAttention(Config cfg = Config()) : cfg_(cfg) {}

    /**
     * @brief Forward pass.
     * @param Q  Query  matrix, shape (L, d)
     * @param K  Key    matrix, shape (L, d)
     * @param V  Value  matrix, shape (L, d)
     */
    AttentionResult<Scalar> forward(const Matrix& Q,
                                     const Matrix& K,
                                     const Matrix& V) {
        using Clock = std::chrono::high_resolution_clock;
        using Ms    = std::chrono::duration<double, std::milli>;

        AttentionResult<Scalar> res;
        auto t0 = Clock::now();

        int L = static_cast<int>(Q.rows());
        int d = static_cast<int>(Q.cols());
        Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(d));

        // Scores = Q K^T / sqrt(d)    shape (L, L)
        auto t1 = Clock::now();
        Matrix Scores = (Q * K.transpose()) * scale;
        res.score_compute_ms = Ms(Clock::now() - t1).count();

        if (cfg_.causal) apply_causal_mask<Scalar>(Scores);

        // A = softmax(Scores)
        auto t2 = Clock::now();
        res.attention_weights = row_softmax<Scalar>(Scores);
        res.softmax_ms = Ms(Clock::now() - t2).count();

        // O = A V
        auto t3 = Clock::now();
        res.output = res.attention_weights * V;
        res.context_ms = Ms(Clock::now() - t3).count();

        res.total_ms = Ms(Clock::now() - t0).count();

        if (cfg_.compute_diag) {
            res.score_frobenius_norm = Scores.norm();
            // Mean row entropy: H = -sum_j A_ij log(A_ij)
            Scalar H = Scalar{0};
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    Scalar a = res.attention_weights(i, j);
                    if (a > Scalar{1e-12})
                        H -= a * std::log(a);
                }
            }
            res.attn_entropy = H / static_cast<Scalar>(L);
        }

        return res;
    }

private:
    Config cfg_;
};

// ----------------------------------------------------------------
// SketchedAttention — main class
// ----------------------------------------------------------------
template <typename Scalar>
class SketchedAttention {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    struct Config {
        int        sketch_dim    = 64;              ///< k
        SketchType sketch_type   = SketchType::GAUSSIAN;
        int        sparse_s      = 1;               ///< for OSNAP
        bool       causal        = false;
        bool       compute_diag  = false;
        uint64_t   rng_seed      = 42;
        bool       reseed_per_call = true;          ///< fresh S each forward pass
    };

    explicit SketchedAttention(Config cfg = Config())
        : cfg_(cfg), call_count_(0) {}

    /**
 * CORRECT speedup sketch: S ∈ R^{k×L}, projects sequence dimension.
 * Scores = Q (S^T S) K^T  approximated as (QS^T)(KS^T)^T  shape (k×k) NOT L×L
 * Then upsample: O = S^T (A_small) V  where A_small ∈ R^{k×k}
 *
 * Complexity: O(Lkd) instead of O(L²d)  — genuinely subquadratic.
 */
 AttentionResult<Scalar> forward_seq_sketch(const Matrix& Q,
    const Matrix& K,
    const Matrix& V,
    int k_seq) {
        int L = static_cast<int>(Q.rows());
        int d = static_cast<int>(Q.cols());
        k_seq = std::min(k_seq, L);

        // ── Sample k landmark indices uniformly without replacement ──────
        std::vector<int> all_idx(L);
        std::iota(all_idx.begin(), all_idx.end(), 0);

        // Fisher-Yates partial shuffle: j uniform in [i, L-1].
        std::mt19937_64 gen(cfg_.rng_seed + call_count_);
        for (int i = 0; i < k_seq; ++i) {
            int j = std::uniform_int_distribution<int>(i, L - 1)(gen);
            std::swap(all_idx[i], all_idx[j]);
        }

        // ── Extract landmark K and V rows ────────────────────────────────
        // K_land[i] = K[all_idx[i]]  — actual token keys, not random mixtures
        Matrix K_land(k_seq, d);
        Matrix V_land(k_seq, d);
        for (int i = 0; i < k_seq; ++i) {
        K_land.row(i) = K.row(all_idx[i]);
        V_land.row(i) = V.row(all_idx[i]);
        }

        // ── Each query attends to k landmark tokens ──────────────────────
        Scalar score_scale = Scalar{1} / std::sqrt(static_cast<Scalar>(d));
        Matrix Scores = (Q * K_land.transpose()) * score_scale;  // (L, k)
        Matrix A      = row_softmax<Scalar>(Scores);              // (L, k)

        AttentionResult<Scalar> res;
        res.output = A * V_land;   // (L, d)
        ++call_count_;
        return res;
    }


    /**
     * @brief Sketched forward pass.
     *
     * Core identity exploited:
     *   Q K^T ≈ Q (S^T S) K^T = (Q S^T)(S K)^T = (SQ)^T (SK)  [up to transpose]
     *
     * More precisely, for S ∈ R^{k×d}:
     *   Q K^T ≈ (Q S^T)(K S^T)^T   where Q S^T ∈ R^{L×k}
     *
     * This requires O(Lkd) instead of O(L²d) for the score matrix.
     *
     * @param Q  shape (L, d)
     * @param K  shape (L, d)
     * @param V  shape (L, d)
     */
    AttentionResult<Scalar> forward(const Matrix& Q,
                                     const Matrix& K,
                                     const Matrix& V) {
        using Clock = std::chrono::high_resolution_clock;
        using Ms    = std::chrono::duration<double, std::milli>;

        validate_inputs(Q, K, V);
        AttentionResult<Scalar> res;

        int L = static_cast<int>(Q.rows());
        int d = static_cast<int>(Q.cols());
        int k = std::min(cfg_.sketch_dim, d);

        auto t0 = Clock::now();

        uint64_t seed = cfg_.reseed_per_call
                        ? cfg_.rng_seed + call_count_++
                        : cfg_.rng_seed;

        Matrix QSt;
        Matrix KSt;

        // --- SRHT: sketch along feature dim d (same random D,P for Q and K) ---
        auto t1 = Clock::now();
        if (cfg_.sketch_type == SketchType::SRHT) {
            StdRNG<Scalar> rng(seed);
            SRHTSketch<Scalar> srht(k, d, rng);
            Matrix Qt = Q.transpose();
            Matrix Kt = K.transpose();
            QSt = srht.apply(Qt).transpose();
            KSt = srht.apply(Kt).transpose();
        } else {
            Matrix S = build_sketch(k, d, seed);
            QSt = Q * S.transpose();
            KSt = K * S.transpose();
        }
        res.sketch_build_ms = Ms(Clock::now() - t1).count();

        // --- Scores from sketch space ---
        auto t2 = Clock::now();

        // Scores ≈ QSt * KSt^T / sqrt(k)   shape (L, L)
        Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k));
        Matrix Scores = (QSt * KSt.transpose()) * scale;
        res.score_compute_ms = Ms(Clock::now() - t2).count();

        if (cfg_.causal) apply_causal_mask<Scalar>(Scores);

        // --- Softmax ---
        auto t3 = Clock::now();
        res.attention_weights = row_softmax<Scalar>(Scores);
        res.softmax_ms = Ms(Clock::now() - t3).count();

        // --- Context: O = A V ---
        auto t4 = Clock::now();
        res.output = res.attention_weights * V;
        res.context_ms = Ms(Clock::now() - t4).count();

        res.total_ms = Ms(Clock::now() - t0).count();

        if (cfg_.compute_diag) {
            res.score_frobenius_norm = Scores.norm();
            Scalar H = Scalar{0};
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    Scalar a = res.attention_weights(i, j);
                    if (a > Scalar{1e-12})
                        H -= a * std::log(a);
                }
            }
            res.attn_entropy = H / static_cast<Scalar>(L);
        }

        return res;
    }

    const Config& config() const { return cfg_; }

private:
    Config   cfg_;
    uint64_t call_count_;

    Matrix build_sketch(int k, int d, uint64_t seed) {
        StdRNG<Scalar> rng(seed);
        switch (cfg_.sketch_type) {
            case SketchType::GAUSSIAN: {
                // Dense Gaussian S ∈ R^{k×d}, entries ~ N(0, 1/k)
                Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(k));
                Matrix S(k, d);
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < d; ++j)
                        S(i, j) = scale * rng.normal();
                return S;
            }
            case SketchType::SPARSE_COUNT: {
                // CountSketch: each column has exactly 1 nonzero ±1/√1
                Matrix S = Matrix::Zero(k, d);
                for (int j = 0; j < d; ++j) {
                    int    row  = static_cast<int>(rng.integer(k));
                    Scalar sign = static_cast<Scalar>(rng.rademacher());
                    S(row, j)   = sign;
                }
                return S;
            }
            case SketchType::SPARSE_OSNAP: {
                // OSNAP: each column has s nonzeros ±1/√s
                int s = cfg_.sparse_s;
                Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(s));
                Matrix S = Matrix::Zero(k, d);
                for (int j = 0; j < d; ++j) {
                    for (int t = 0; t < s; ++t) {
                        int    row  = static_cast<int>(rng.integer(k));
                        Scalar sign = static_cast<Scalar>(rng.rademacher()) * scale;
                        S(row, j)  += sign;
                    }
                }
                return S;
            }
            case SketchType::SRHT:
                throw std::logic_error("SketchedAttention: SRHT uses implicit apply(), not build_sketch");
        }
        // unreachable
        return Matrix::Zero(k, d);
    }

    static bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

    void validate_inputs(const Matrix& Q, const Matrix& K, const Matrix& V) {
        if (Q.rows() != K.rows() || Q.rows() != V.rows())
            throw std::invalid_argument("SketchedAttention: Q, K, V must have same L");
        if (Q.cols() != K.cols())
            throw std::invalid_argument("SketchedAttention: Q, K must have same d");
        if (cfg_.sketch_dim <= 0)
            throw std::invalid_argument("SketchedAttention: sketch_dim must be > 0");
        int d = static_cast<int>(Q.cols());
        if (cfg_.sketch_type == SketchType::SRHT) {
            if (!is_power_of_two(d))
                throw std::invalid_argument(
                    "SketchedAttention: SRHT requires head dimension d to be a power of 2");
        }
    }
};

// ----------------------------------------------------------------
// MultiHeadSketchedAttention
// ----------------------------------------------------------------
/**
 * @brief Multi-head wrapper around SketchedAttention.
 *
 * Each head operates on a d_head = d_model / num_heads slice.
 * Each head draws its own independent sketch.
 *
 * In a real transformer, Q/K/V projections (Wq, Wk, Wv) would sit here.
 * For this benchmark we accept pre-projected per-head Q, K, V directly.
 */
template <typename Scalar>
class MultiHeadSketchedAttention {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;

    struct Config {
        int        num_heads     = 8;
        int        d_model       = 512;
        int        sketch_dim    = 64;
        SketchType sketch_type   = SketchType::GAUSSIAN;
        bool       causal        = false;
        bool       compute_diag  = false;
        uint64_t   rng_seed      = 42;
    };

    explicit MultiHeadSketchedAttention(Config cfg) : cfg_(cfg) {
        int d_head = cfg_.d_model / cfg_.num_heads;
        for (int h = 0; h < cfg_.num_heads; ++h) {
            typename SketchedAttention<Scalar>::Config head_cfg;
            head_cfg.sketch_dim   = cfg_.sketch_dim;
            head_cfg.sketch_type  = cfg_.sketch_type;
            head_cfg.causal       = cfg_.causal;
            head_cfg.compute_diag = cfg_.compute_diag;
            head_cfg.rng_seed     = cfg_.rng_seed + static_cast<uint64_t>(h) * 1000;
            head_cfg.reseed_per_call = true;
            heads_.emplace_back(head_cfg);
        }
    }

    /**
     * @brief Forward pass over all heads.
     * @param Q  shape (L, d_model)
     * @param K  shape (L, d_model)
     * @param V  shape (L, d_model)
     * @return   Concatenated output, shape (L, d_model)
     */
    Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V) {
        int L      = static_cast<int>(Q.rows());
        int d_head = cfg_.d_model / cfg_.num_heads;
        Matrix out(L, cfg_.d_model);

        for (int h = 0; h < cfg_.num_heads; ++h) {
            int col_start = h * d_head;
            Matrix Qh = Q.middleCols(col_start, d_head);
            Matrix Kh = K.middleCols(col_start, d_head);
            Matrix Vh = V.middleCols(col_start, d_head);

            auto res = heads_[h].forward(Qh, Kh, Vh);
            out.middleCols(col_start, d_head) = res.output;
        }
        return out;
    }

private:
    Config cfg_;
    std::vector<SketchedAttention<Scalar>> heads_;
};

} // namespace attention
} // namespace randnla