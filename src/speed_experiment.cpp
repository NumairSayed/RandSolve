/**
 * @file speed_experiment.cpp
 * @brief Benchmarking harness comparing QRSolver vs SketchedLeastSquares.
 *
 * Methodology:
 *   - Generate a tall random matrix A ∈ R^{m×n}  (m >> n)
 *   - Solve with QR, NormalEq, and GaussianSketch
 *   - Measure wall time using std::chrono::high_resolution_clock
 *   - Report residual norm and speedup factor
 *
 * Compile and run:
 *   ./speed_experiment [m] [n] [sketch_factor]
 */

#include "core/Precision.hpp"
#include "rng/StdRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "solvers/QRSolver.hpp"
#include "solvers/NormalEquationSolver.hpp"
#include "solvers/IterativeSolver.hpp"
#include "randomized/SketchedLeastSquares.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>
#include <functional>

using Scalar  = double;
using Matrix  = randnla::PrecisionTraits<Scalar>::Matrix;
using Vector  = randnla::PrecisionTraits<Scalar>::Vector;
using Clock   = std::chrono::high_resolution_clock;
using Ms      = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// Timer helper
// ----------------------------------------------------------------
struct TimedResult {
    Vector  solution;
    double  elapsed_ms;
    Scalar  residual_norm;
};

template <typename Fn>
TimedResult timed_run(Fn&& fn, const Matrix& A, const Vector& b) {
    auto t0 = Clock::now();
    Vector x = fn();
    auto t1 = Clock::now();
    double elapsed = Ms(t1 - t0).count();
    Scalar res = (A * x - b).norm();
    return {x, elapsed, res};
}

// ----------------------------------------------------------------
// Print table row
// ----------------------------------------------------------------
void print_row(const std::string& name,
               double elapsed_ms,
               double residual,
               double speedup) {
    std::cout << std::left  << std::setw(32) << name
              << std::right << std::setw(12) << std::scientific
                            << std::setprecision(3) << residual
              << std::setw(12) << std::fixed << std::setprecision(1)
                            << elapsed_ms << " ms"
              << std::setw(10) << std::setprecision(2) << speedup << "x"
              << "\n";
}

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
    // Parse optional arguments
    int    m             = (argc > 1) ? std::stoi(argv[1]) : 10000;
    int    n             = (argc > 2) ? std::stoi(argv[2]) : 100;
    int    sketch_factor = (argc > 3) ? std::stoi(argv[3]) : 6;
    int    k             = sketch_factor * n;
    int    runs          = 3;  // average over multiple runs

    // Clamp
    k = std::min(k, m);

    std::cout << "\n";
    std::cout << "====================================================\n";
    std::cout << "  RandNLA Speed Experiment\n";
    std::cout << "====================================================\n";
    std::cout << "  Matrix size    : " << m << " x " << n << "\n";
    std::cout << "  Sketch dim k   : " << k
              << "  (factor " << sketch_factor << "x)\n";
    std::cout << "  Repetitions    : " << runs << "\n";
    std::cout << "====================================================\n\n";

    // Generate problem
    Matrix A = Matrix::Random(m, n);
    // Low-condition-number construction: A = U * sigma * V^T
    // For simplicity, use random A (condition ~sqrt(m/n) for Gaussian)
    Vector x_true = Vector::Random(n);
    Vector b      = A * x_true + 1e-3 * Vector::Random(m);

    // ---- Warm-up (avoid cold cache effects) ----
    {
        randnla::QRSolver<Scalar> s;
        volatile auto tmp = s.solve(A, b).norm();
        (void)tmp;
    }

    // ---- Run each solver `runs` times, take minimum elapsed ----
    struct Entry {
        std::string name;
        double      best_ms  = 1e18;
        Scalar      residual = 0;
    };

    std::vector<Entry> entries;

    auto bench = [&](const std::string& name, std::function<Vector()> fn) {
        Entry e;
        e.name = name;
        for (int r = 0; r < runs; ++r) {
            auto res = timed_run([&]{ return fn(); }, A, b);
            if (res.elapsed_ms < e.best_ms) {
                e.best_ms  = res.elapsed_ms;
                e.residual = res.residual_norm;
            }
        }
        entries.push_back(e);
    };

    // QR
    bench("QR (HouseholderQR)", [&]() {
        randnla::QRSolver<Scalar> s;
        return s.solve(A, b);
    });

    // Normal Equations
    bench("Normal Equations (LDLT)", [&]() {
        randnla::NormalEquationSolver<Scalar> s;
        return s.solve(A, b);
    });

    // Iterative CG
    bench("Iterative CG", [&]() {
        randnla::IterativeSolver<Scalar> s(0, 1e-8);
        return s.solve(A, b);
    });

    // Gaussian Sketch
    bench("GaussianSketch + QR  k=" + std::to_string(k), [&]() {
        randnla::StdRNG<Scalar> rng(42);
        auto sketch = std::make_shared<randnla::GaussianSketch<Scalar>>(k, m, rng);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        return sls.solve(A, b);
    });

    // Sparse Sketch (CountSketch)
    bench("SparseSketch(s=1) + QR k=" + std::to_string(k), [&]() {
        randnla::StdRNG<Scalar> rng(42);
        auto sketch = std::make_shared<randnla::SparseSketch<Scalar>>(k, m, rng, 1);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        return sls.solve(A, b);
    });

    // Sparse Sketch (OSNAP s=4)
    bench("SparseSketch(s=4) + QR k=" + std::to_string(k), [&]() {
        randnla::StdRNG<Scalar> rng(42);
        int s = std::min(4, k);
        auto sketch = std::make_shared<randnla::SparseSketch<Scalar>>(k, m, rng, s);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        return sls.solve(A, b);
    });

    // ---- Print results ----
    double baseline_ms = entries[0].best_ms;

    std::cout << std::left  << std::setw(32) << "Method"
              << std::right << std::setw(12) << "Residual"
              << std::setw(12) << "Time"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (const auto& e : entries) {
        double speedup = baseline_ms / e.best_ms;
        print_row(e.name, e.best_ms, static_cast<double>(e.residual), speedup);
    }

    std::cout << "\n";
    std::cout << "Notes:\n";
    std::cout << "  • Speedup is relative to QR (HouseholderQR baseline).\n";
    std::cout << "  • Times are minimums over " << runs << " runs.\n";
    std::cout << "  • For GaussianSketch, sketch construction time is included.\n";
    std::cout << "  • To change problem size: ./speed_experiment <m> <n> <factor>\n";
    std::cout << "====================================================\n\n";

    return 0;
}