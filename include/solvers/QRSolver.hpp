#pragma once
/**
 * @file QRSolver.hpp
 * @brief Least-squares solver via Householder QR decomposition.
 *
 * Uses Eigen's HouseholderQR for full-rank systems.
 * For rank-deficient systems, ColPivHouseholderQR is used as a fallback.
 *
 * Complexity: O(mn^2) for a (m×n) matrix.
 *
 * @tparam Scalar  float or double.
 */

#include "ILeastSquaresSolver.hpp"
#include <Eigen/QR>
#include <stdexcept>

namespace randnla {

template <typename Scalar>
class QRSolver : public ILeastSquaresSolver<Scalar> {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    enum class Variant {
        HOUSEHOLDER,      ///< Standard HouseholderQR (assumes full column rank)
        COL_PIVOT,        ///< ColPivHouseholderQR    (rank-revealing)
        FULL_PIVOT,       ///< FullPivHouseholderQR   (most robust, slower)
    };

    explicit QRSolver(Variant v = Variant::HOUSEHOLDER) : variant_(v) {}

    Vector solve(const Matrix& A, const Vector& b) override {
        if (A.rows() < A.cols())
            throw std::invalid_argument("QRSolver: system must be overdetermined (m >= n)");
        switch (variant_) {
            case Variant::HOUSEHOLDER:
                return A.householderQr().solve(b);
            case Variant::COL_PIVOT:
                return A.colPivHouseholderQr().solve(b);
            case Variant::FULL_PIVOT:
                return A.fullPivHouseholderQr().solve(b);
        }
        // unreachable
        return A.householderQr().solve(b);
    }

private:
    Variant variant_;
};

} // namespace randnla