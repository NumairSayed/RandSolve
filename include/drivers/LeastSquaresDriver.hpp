#pragma once
/**
 * @file LeastSquaresDriver.hpp
 * @brief High-level driver that routes a least-squares problem to the
 *        appropriate algorithm based on matrix properties.
 *
 * Usage:
 * @code
 *   LeastSquaresDriver<double> driver;
 *   auto x = driver.solve(A, b);
 * @endcode
 *
 * The driver uses DecisionEngine internally and constructs concrete objects
 * with sensible defaults.  For expert control, bypass the driver and
 * construct algorithms directly.
 *
 * @tparam Scalar  float or double.
 */

#include "core/Precision.hpp"
#include "core/MatrixProperties.hpp"
#include "core/DecisionEngine.hpp"
#include "rng/StdRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "solvers/QRSolver.hpp"
#include "solvers/NormalEquationSolver.hpp"
#include "solvers/IterativeSolver.hpp"
#include "randomized/SketchedLeastSquares.hpp"
#include "randomized/PreconditionedLeastSquares.hpp"

#include <memory>
#include <iostream>

namespace randnla {

template <typename Scalar>
class LeastSquaresDriver {
public:
    using Matrix = typename PrecisionTraits<Scalar>::Matrix;
    using Vector = typename PrecisionTraits<Scalar>::Vector;

    struct Options {
        bool   verbose       = false;  ///< Print routing decision
        int    sketch_factor = 4;      ///< k = sketch_factor * n
        uint64_t rng_seed    = 42;
        DecisionEngine::Config engine_config;
    };

    explicit LeastSquaresDriver(Options opts = {})
        : opts_(opts), engine_(opts.engine_config) {}

    Vector solve(const Matrix& A, const Vector& b,
                 MatrixProperties props = {}) {
        // Fill in basic properties if not provided
        props.m = A.rows();
        props.n = A.cols();

        AlgorithmChoice choice = engine_.route(props);

        if (opts_.verbose)
            std::cout << "[LeastSquaresDriver] Routing to: "
                      << to_string(choice) << "\n";

        switch (choice) {
            case AlgorithmChoice::QR_DIRECT: {
                QRSolver<Scalar> solver;
                return solver.solve(A, b);
            }
            case AlgorithmChoice::NORMAL_EQUATIONS: {
                NormalEquationSolver<Scalar> solver;
                return solver.solve(A, b);
            }
            case AlgorithmChoice::ITERATIVE_CG: {
                auto solver = std::make_shared<IterativeSolver<Scalar>>();
                return solver->solve(A, b);
            }
            case AlgorithmChoice::SKETCHED_LS: {
                int k = std::max(static_cast<int>(A.cols()) * opts_.sketch_factor,
                                 static_cast<int>(A.cols()) + 1);
                k = std::min(k, static_cast<int>(A.rows()));
                rng_ = std::make_unique<StdRNG<Scalar>>(opts_.rng_seed);
                auto sketch = std::make_shared<GaussianSketch<Scalar>>(
                    k, static_cast<int>(A.rows()), *rng_);
                auto solver = std::make_shared<QRSolver<Scalar>>();
                SketchedLeastSquares<Scalar> sls(sketch, solver);
                return sls.solve(A, b);
            }
            case AlgorithmChoice::PRECONDITIONED_LS: {
                int k = std::max(static_cast<int>(A.cols()) * opts_.sketch_factor,
                                 static_cast<int>(A.cols()) + 1);
                k = std::min(k, static_cast<int>(A.rows()));
                rng_ = std::make_unique<StdRNG<Scalar>>(opts_.rng_seed);
                auto sketch = std::make_shared<GaussianSketch<Scalar>>(
                    k, static_cast<int>(A.rows()), *rng_);
                auto iter_solver = std::make_shared<IterativeSolver<Scalar>>();
                PreconditionedLeastSquares<Scalar> pls(sketch, iter_solver);
                return pls.solve(A, b);
            }
            case AlgorithmChoice::RANDOMIZED_SVD:
                // Fall through to QR for now; low-rank path is in LowRankDriver.
                [[fallthrough]];
            default: {
                QRSolver<Scalar> solver;
                return solver.solve(A, b);
            }
        }
    }

private:
    Options                          opts_;
    DecisionEngine                   engine_;
    std::unique_ptr<StdRNG<Scalar>>  rng_;
};

} // namespace randnla