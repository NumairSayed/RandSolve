#pragma once
/**
 * @file ISketch.hpp
 * @brief Abstract sketching operator interface (RandBLAS-inspired).
 *
 * A sketch S ∈ R^{k×m} is an oblivious linear map that compresses
 * A ∈ R^{m×n} to SA ∈ R^{k×n} (and b to Sb ∈ R^k) while approximately
 * preserving norms and inner products when k is chosen appropriately.
 *
 * Design constraints:
 *   - ISketch depends on IRNG only, never on solver internals.
 *   - The operator is applied lazily (no explicit S matrix stored unless needed).
 *   - Subclasses must document their embedding dimension guarantees.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"

namespace randnla {

template <typename Scalar>
class ISketch {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    virtual ~ISketch() = default;

    /**
     * @brief Apply the sketching operator to matrix A.
     * @param A  Input matrix of shape (m, n).
     * @return   Sketched matrix of shape (k, n).
     */
    virtual Matrix apply(const Matrix& A) = 0;

    /**
     * @brief Apply the sketching operator to vector b.
     * @param b  Input vector of length m.
     * @return   Sketched vector of length k.
     */
    virtual Vector apply(const Vector& b) = 0;

    /// Target sketch dimension k.
    virtual int sketch_dim() const = 0;

    /// Input dimension m that this sketch was configured for.
    virtual int input_dim() const = 0;
};

} // namespace randnla