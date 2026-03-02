/**
 * @file main.cpp
 * @brief Minimal working demonstration of the RandNLA framework.
 *
 * Demonstrates:
 *   1. Direct construction: GaussianSketch + QRSolver → SketchedLeastSquares
 *   2. High-level driver: LeastSquaresDriver with automatic routing
 *   3. Low-rank approximation via RandomizedSVD
 *   4. Confidence bound estimation
 */

#include "core/Precision.hpp"
#include "core/MatrixProperties.hpp"
#include "core/DecisionEngine.hpp"
#include "rng/StdRNG.hpp"
#include "rng/EigenRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "solvers/QRSolver.hpp"
#include "solvers/NormalEquationSolver.hpp"
#include "solvers/IterativeSolver.hpp"
#include "randomized/SketchedLeastSquares.hpp"
#include "randomized/ConfidenceBounds.hpp"
#include "lowrank/RandomizedSVD.hpp"
#include "drivers/LeastSquaresDriver.hpp"
#include "drivers/LowRankDriver.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <memory>

using Scalar = double;
using Traits = randnla::PrecisionTraits<Scalar>;
using Matrix = Traits::Matrix;
using Vector = Traits::Vector;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "======================================\n";
    std::cout << "  RandNLA C++ Demo\n";
    std::cout << "======================================\n\n";

    // ----------------------------------------------------------------
    // 1. Generate a tall overdetermined system  A ∈ R^{500×30}
    // ----------------------------------------------------------------
    const int m = 500, n = 30;
    Matrix A = Matrix::Random(m, n);
    Vector x_true = Vector::Random(n);
    Vector b = A * x_true + 0.01 * Vector::Random(m);  // noisy RHS

    std::cout << "Problem: " << m << " x " << n
              << "  (overdetermined, condition ~1)\n\n";

    // ----------------------------------------------------------------
    // 2. Direct QR solve (ground truth)
    // ----------------------------------------------------------------
    {
        randnla::QRSolver<Scalar> solver;
        Vector x_qr = solver.solve(A, b);
        Scalar res = (A * x_qr - b).norm();
        std::cout << "[QRSolver]            residual = " << res << "\n";
    }

    // ----------------------------------------------------------------
    // 3. Normal Equations
    // ----------------------------------------------------------------
    {
        randnla::NormalEquationSolver<Scalar> solver;
        Vector x_ne = solver.solve(A, b);
        Scalar res = (A * x_ne - b).norm();
        std::cout << "[NormalEqSolver]      residual = " << res << "\n";
    }

    // ----------------------------------------------------------------
    // 4. Sketch-and-solve: GaussianSketch + QRSolver
    // ----------------------------------------------------------------
    {
        const int k = 4 * n;  // sketch dimension
        randnla::StdRNG<Scalar> rng(42);
        auto sketch = std::make_shared<randnla::GaussianSketch<Scalar>>(k, m, rng);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        Vector x_sls = sls.solve(A, b);
        Scalar res = (A * x_sls - b).norm();
        std::cout << "[SketchedLS/Gaussian] residual = " << res
                  << "  (k=" << k << ")\n";

        // Confidence bound
        auto bound = randnla::ConfidenceBounds<Scalar>::analytic_bound(
            A, b, x_sls, 0.5);
        std::cout << "  → relative_error      = " << bound.relative_error << "\n";
        std::cout << "  → probabilistic_bound = " << bound.probabilistic_bound << "\n";
    }

    // ----------------------------------------------------------------
    // 5. Sketch-and-solve: SparseSketch
    // ----------------------------------------------------------------
    {
        const int k = 6 * n;
        randnla::StdRNG<Scalar> rng(123);
        auto sketch = std::make_shared<randnla::SparseSketch<Scalar>>(k, m, rng, 1);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        Vector x_sls = sls.solve(A, b);
        Scalar res = (A * x_sls - b).norm();
        std::cout << "[SketchedLS/Sparse]   residual = " << res
                  << "  (k=" << k << ", s=1)\n";
    }

    // ----------------------------------------------------------------
    // 6. Iterative CG Solver
    // ----------------------------------------------------------------
    {
        randnla::IterativeSolver<Scalar> solver(500, 1e-8);
        Vector x_cg = solver.solve(A, b);
        Scalar res = (A * x_cg - b).norm();
        std::cout << "[IterativeSolver/CG]  residual = " << res
                  << "  (iters=" << solver.last_iterations() << ")\n";
    }

    // ----------------------------------------------------------------
    // 7. High-level driver (auto-routing)
    // ----------------------------------------------------------------
    {
        std::cout << "\n--- LeastSquaresDriver (auto-route) ---\n";
        typename randnla::LeastSquaresDriver<Scalar>::Options opts;
        opts.verbose = true;
        opts.sketch_factor = 4;
        randnla::LeastSquaresDriver<Scalar> driver(opts);

        // a) normal problem
        Vector x_d = driver.solve(A, b);
        std::cout << "  residual (auto) = " << (A * x_d - b).norm() << "\n";

        // b) force tall route
        randnla::MatrixProperties props;
        props.m = m; props.n = n;
        props.density = 1.0;
        props.condition_number = 1.0;
        Vector x_d2 = driver.solve(A, b, props);
        std::cout << "  residual (props) = " << (A * x_d2 - b).norm() << "\n";
    }

    // ----------------------------------------------------------------
    // 8. Randomized SVD
    // ----------------------------------------------------------------
    {
        std::cout << "\n--- RandomizedSVD ---\n";
        const int m2 = 200, n2 = 100, r = 5;
        // Construct rank-5 matrix
        Matrix A_lr = Matrix::Random(m2, r) * Matrix::Random(r, n2);
        A_lr += 0.001 * Matrix::Random(m2, n2);

        randnla::StdRNG<Scalar> rng(7);
        randnla::RandomizedSVD<Scalar> rsvd(10, 2);
        rsvd.compute(A_lr, r, rng);

        Scalar err = (A_lr - rsvd.reconstruct()).norm() / A_lr.norm();
        std::cout << "  Rank-" << r << " approx relative error: " << err << "\n";
        std::cout << "  Top-5 singular values: "
                  << rsvd.singularValues().transpose() << "\n";
    }

    // ----------------------------------------------------------------
    // 9. Decision Engine demo
    // ----------------------------------------------------------------
    {
        std::cout << "\n--- DecisionEngine ---\n";
        randnla::DecisionEngine engine;
        auto demo = [&](const char* label, randnla::MatrixProperties p) {
            std::cout << "  " << label << " → "
                      << randnla::to_string(engine.route(p)) << "\n";
        };
        demo("small 100×10",
             {100, 10, 1.0, 0, 1.0});
        randnla::MatrixProperties tall; tall.m=50000; tall.n=20;
        demo("tall 50000×20", tall);
        randnla::MatrixProperties lr;   lr.m=1000; lr.n=1000; lr.estimated_rank=5;
        demo("lowrank 1000×1000 r=5", lr);
        randnla::MatrixProperties illcond;
        illcond.m=1000; illcond.n=50; illcond.condition_number=1e14;
        demo("ill-cond 1000×50", illcond);
    }

    std::cout << "\n======================================\n";
    std::cout << "  All demos completed successfully.\n";
    std::cout << "======================================\n";
    return 0;
}