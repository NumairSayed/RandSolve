/**
 * @file gauss_newton_experiment.cpp
 * @brief Research-grade experimental harness comparing:
 *        Classical QR vs Randomized Sketch-and-Solve
 *        inside repeated Gauss-Newton iterations.
 *
 * Problem:
 *   min_x || f(x) ||^2
 *   f_i(x) = sin(a_i^T x) - b_i,   i = 1..m
 *   J_ij(x) = cos(a_i^T x) * a_ij        (Jacobian)
 *
 * Gauss-Newton step:
 *   Solve  min_delta || J(x) * delta + r(x) ||^2
 *   x <- x + delta
 *
 * Metrics collected per iteration:
 *   - Wall time for linear solve
 *   - Residual norm || f(x) ||
 *   - Step norm || delta ||
 *   - Relative residual change
 *
 * Usage:
 *   ./gauss_newton_experiment [m] [n] [max_iter] [tol]
 *   ./gauss_newton_experiment 50000 100 50 1e-6
 *
 * Sketch sizes tested: k = {2n, 4n, 6n, 8n, full QR}
 */

#include "core/Precision.hpp"
#include "rng/StdRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "solvers/QRSolver.hpp"
#include "solvers/NormalEquationSolver.hpp"
#include "randomized/SketchedLeastSquares.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <memory>
#include <functional>
#include <cassert>
#include <numeric>
#include <algorithm>

// ============================================================
// Types
// ============================================================
using Scalar  = double;
using Matrix  = randnla::PrecisionTraits<Scalar>::Matrix;
using Vector  = randnla::PrecisionTraits<Scalar>::Vector;
using Clock   = std::chrono::high_resolution_clock;
using Ms      = std::chrono::duration<double, std::milli>;

// ============================================================
// Per-iteration record
// ============================================================
struct IterRecord {
    int    iter;
    double residual_norm;       // || f(x) ||_2
    double relative_residual;   // || f(x) || / || f(x_0) ||
    double step_norm;           // || delta ||_2
    double solve_time_ms;       // time spent in linear solve only
    double total_iter_time_ms;  // full iteration (J build + solve)
    bool   converged;
};

// ============================================================
// Full run result
// ============================================================
struct RunResult {
    std::string              label;
    std::vector<IterRecord>  iters;
    double                   total_time_ms;
    bool                     converged;
    int                      iters_to_converge;
    double                   final_residual;

    // Timing breakdown
    double total_solve_ms   = 0.0;
    double total_jacobian_ms= 0.0;
};

// ============================================================
// Problem definition
// ============================================================
struct Problem {
    int    m, n;
    Matrix A;      // shape (m, n) — the a_i vectors stacked as rows
    Vector b;      // shape (m,)
    Vector x_true; // ground truth (for reference)

    /**
     * Residual vector f(x), shape (m,).
     * f_i(x) = sin(a_i^T x) - b_i
     */
    Vector residual(const Vector& x) const {
        Vector Ax = A * x;          // (m,)
        Vector r(m);
        for (int i = 0; i < m; ++i)
            r(i) = std::sin(Ax(i)) - b(i);
        return r;
    }

    /**
     * Jacobian J(x), shape (m, n).
     * J_ij = cos(a_i^T x) * a_ij
     */
    Matrix jacobian(const Vector& x) const {
        Vector Ax = A * x;          // (m,)
        Vector c(m);
        for (int i = 0; i < m; ++i)
            c(i) = std::cos(Ax(i));
        // J = diag(c) * A
        return c.asDiagonal() * A;
    }
};

// ============================================================
// Generate synthetic problem
// ============================================================
Problem make_problem(int m, int n, uint64_t seed = 42) {
    Problem p;
    p.m = m;
    p.n = n;

    std::mt19937_64 gen(seed);
    std::normal_distribution<Scalar> nd(0.0, 1.0);
    std::uniform_real_distribution<Scalar> ud(-1.0, 1.0);

    // a_i ~ N(0, 1/n * I)  — scale so a_i^T x is O(1)
    Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(n));
    p.A.resize(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            p.A(i, j) = scale * nd(gen);

    // x_true ~ U(-1, 1)
    p.x_true.resize(n);
    for (int j = 0; j < n; ++j)
        p.x_true(j) = ud(gen);

    // b_i = sin(a_i^T x_true) + noise
    Vector Ax_true = p.A * p.x_true;
    p.b.resize(m);
    std::normal_distribution<Scalar> noise(0.0, 0.01);
    for (int i = 0; i < m; ++i)
        p.b(i) = std::sin(Ax_true(i)) + noise(gen);

    return p;
}

// ============================================================
// Linear solver abstraction for the GN step
// ============================================================
using LinearSolver = std::function<Vector(const Matrix&, const Vector&, double&)>;
// Returns delta, writes solve_time_ms via reference

// --- QR solver ---
LinearSolver make_qr_solver() {
    return [](const Matrix& J, const Vector& r, double& solve_ms) -> Vector {
        auto t0 = Clock::now();
        randnla::QRSolver<Scalar> solver;
        Vector delta = solver.solve(J, -r);   // solve J*delta = -r
        solve_ms = Ms(Clock::now() - t0).count();
        return delta;
    };
}

// --- Normal equations solver ---
LinearSolver make_normal_solver() {
    return [](const Matrix& J, const Vector& r, double& solve_ms) -> Vector {
        auto t0 = Clock::now();
        randnla::NormalEquationSolver<Scalar> solver;
        Vector delta = solver.solve(J, -r);
        solve_ms = Ms(Clock::now() - t0).count();
        return delta;
    };
}

// --- Sketched solver: Gaussian ---
LinearSolver make_gaussian_sketch_solver(int k, uint64_t seed = 99) {
    return [k, seed](const Matrix& J, const Vector& r, double& solve_ms) -> Vector {
        auto t0 = Clock::now();
        int m = static_cast<int>(J.rows());
        randnla::StdRNG<Scalar> rng(seed);
        auto sketch = std::make_shared<randnla::GaussianSketch<Scalar>>(k, m, rng);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        Vector delta = sls.solve(J, -r);
        solve_ms = Ms(Clock::now() - t0).count();
        return delta;
    };
}

// --- Sketched solver: Sparse (CountSketch) ---
LinearSolver make_sparse_sketch_solver(int k, int s, uint64_t seed = 99) {
    return [k, s, seed](const Matrix& J, const Vector& r, double& solve_ms) -> Vector {
        auto t0 = Clock::now();
        int m = static_cast<int>(J.rows());
        randnla::StdRNG<Scalar> rng(seed);
        auto sketch = std::make_shared<randnla::SparseSketch<Scalar>>(k, m, rng, s);
        auto solver = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(sketch, solver);
        Vector delta = sls.solve(J, -r);
        solve_ms = Ms(Clock::now() - t0).count();
        return delta;
    };
}

// ============================================================
// Gauss-Newton loop
// ============================================================
RunResult run_gauss_newton(
    const Problem&       prob,
    const Vector&        x0,
    LinearSolver         solver,
    const std::string&   label,
    int                  max_iter,
    Scalar               tol,
    bool                 verbose)
{
    RunResult result;
    result.label = label;
    result.converged = false;
    result.iters_to_converge = max_iter;

    Vector x = x0;
    Scalar r0_norm = prob.residual(x0).norm();

    auto wall_start = Clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {
        auto iter_start = Clock::now();

        // --- Build residual and Jacobian ---
        auto t_jac = Clock::now();
        Vector r = prob.residual(x);
        Matrix J = prob.jacobian(x);
        double jac_ms = Ms(Clock::now() - t_jac).count();

        Scalar r_norm = r.norm();
        Scalar rel    = r_norm / (r0_norm + Scalar{1e-300});

        // --- Linear solve: min || J*delta + r ||^2 ---
        double solve_ms = 0.0;
        Vector delta = solver(J, r, solve_ms);

        // --- Update ---
        x += delta;

        double iter_ms = Ms(Clock::now() - iter_start).count();

        IterRecord rec;
        rec.iter               = iter + 1;
        rec.residual_norm      = r_norm;
        rec.relative_residual  = rel;
        rec.step_norm          = delta.norm();
        rec.solve_time_ms      = solve_ms;
        rec.total_iter_time_ms = iter_ms;
        rec.converged          = (r_norm < tol);

        result.iters.push_back(rec);
        result.total_solve_ms    += solve_ms;
        result.total_jacobian_ms += jac_ms;

        if (verbose) {
            std::cout << "  [" << std::setw(3) << rec.iter << "]"
                      << "  ||r|| = " << std::scientific << std::setprecision(4)
                      << r_norm
                      << "  ||delta|| = " << delta.norm()
                      << "  solve = " << std::fixed << std::setprecision(1)
                      << solve_ms << " ms\n";
        }

        if (r_norm < tol) {
            result.converged = true;
            result.iters_to_converge = iter + 1;
            break;
        }
    }

    result.total_time_ms  = Ms(Clock::now() - wall_start).count();
    result.final_residual = result.iters.back().residual_norm;
    return result;
}

// ============================================================
// Print comparison table
// ============================================================
void print_summary_table(const std::vector<RunResult>& results) {
    const int W = 32;
    std::cout << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "  SUMMARY TABLE\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left  << std::setw(W)   << "Method"
              << std::right << std::setw(10)  << "Converged"
              << std::setw(8)   << "Iters"
              << std::setw(16)  << "Final ||r||"
              << std::setw(16)  << "Total (ms)"
              << std::setw(16)  << "Solve (ms)"
              << std::setw(16)  << "Jacobian (ms)"
              << std::setw(10)  << "Speedup"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    double baseline_total = -1.0;
    double baseline_solve = -1.0;
    for (const auto& r : results) {
        if (r.label.find("QR") != std::string::npos && baseline_total < 0) {
            baseline_total = r.total_time_ms;
            baseline_solve = r.total_solve_ms;
        }
    }

    for (const auto& r : results) {
        double speedup = (baseline_total > 0) ? baseline_total / r.total_time_ms : 1.0;
        std::cout << std::left  << std::setw(W)   << r.label
                  << std::right << std::setw(10)  << (r.converged ? "YES" : "NO")
                  << std::setw(8)   << r.iters_to_converge
                  << std::setw(16)  << std::scientific << std::setprecision(4)
                                    << r.final_residual
                  << std::setw(16)  << std::fixed << std::setprecision(1)
                                    << r.total_time_ms
                  << std::setw(16)  << r.total_solve_ms
                  << std::setw(16)  << r.total_jacobian_ms
                  << std::setw(10)  << std::setprecision(2) << speedup << "x"
                  << "\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

// ============================================================
// Print per-iteration residual decay for one method
// ============================================================
void print_residual_trace(const RunResult& r) {
    std::cout << "\n  Residual decay — " << r.label << "\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    std::cout << "  " << std::left << std::setw(8) << "Iter"
              << std::right << std::setw(16) << "||r||"
              << std::setw(14) << "Rel ||r||"
              << std::setw(14) << "||delta||"
              << std::setw(14) << "Solve(ms)"
              << "\n";
    std::cout << "  " << std::string(70, '-') << "\n";
    for (const auto& rec : r.iters) {
        std::cout << "  " << std::left  << std::setw(8)  << rec.iter
                  << std::right << std::setw(16) << std::scientific
                                << std::setprecision(4) << rec.residual_norm
                  << std::setw(14) << rec.relative_residual
                  << std::setw(14) << rec.step_norm
                  << std::setw(14) << std::fixed << std::setprecision(2)
                                << rec.solve_time_ms
                  << "\n";
    }
}

// ============================================================
// Export results to CSV for plotting
// ============================================================
void export_csv(const std::vector<RunResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[WARN] Could not open " << filename << " for CSV export\n";
        return;
    }
    f << "method,iter,residual_norm,relative_residual,step_norm,"
      << "solve_time_ms,total_iter_time_ms\n";
    for (const auto& r : results) {
        for (const auto& rec : r.iters) {
            f << r.label << ","
              << rec.iter << ","
              << std::scientific << std::setprecision(8)
              << rec.residual_norm << ","
              << rec.relative_residual << ","
              << rec.step_norm << ","
              << std::fixed << std::setprecision(4)
              << rec.solve_time_ms << ","
              << rec.total_iter_time_ms << "\n";
        }
    }
    std::cout << "[CSV] Results written to " << filename << "\n";
}

// ============================================================
// Sketch-size sensitivity study
// ============================================================
void sketch_size_study(const Problem& prob,
                       const Vector& x0,
                       int max_iter,
                       Scalar tol)
{
    std::cout << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "  SKETCH SIZE SENSITIVITY STUDY  (n=" << prob.n << ")\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << std::left  << std::setw(28) << "Sketch dim k"
              << std::right << std::setw(10) << "k/n ratio"
              << std::setw(10) << "Converged"
              << std::setw(8)  << "Iters"
              << std::setw(16) << "Final ||r||"
              << std::setw(16) << "Total (ms)"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    // QR baseline
    auto qr_result = run_gauss_newton(
        prob, x0, make_qr_solver(), "QR baseline", max_iter, tol, false);
    double baseline_ms = qr_result.total_time_ms;

    std::vector<int> factors = {2, 3, 4, 6, 8, 12, 16};
    for (int f : factors) {
        int k = std::min(f * prob.n, prob.m);
        std::string lbl = "GaussSketch k=" + std::to_string(k)
                          + " (" + std::to_string(f) + "n)";
        auto res = run_gauss_newton(
            prob, x0, make_gaussian_sketch_solver(k), lbl, max_iter, tol, false);
        double speedup = baseline_ms / res.total_time_ms;
        std::cout << std::left  << std::setw(28) << lbl
                  << std::right << std::setw(10) << std::fixed
                                << std::setprecision(1) << Scalar(k) / prob.n
                  << std::setw(10) << (res.converged ? "YES" : "NO")
                  << std::setw(8)  << res.iters_to_converge
                  << std::setw(16) << std::scientific << std::setprecision(4)
                                   << res.final_residual
                  << std::setw(16) << std::fixed << std::setprecision(1)
                                   << res.total_time_ms
                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                  << "\n";
    }
    // Print QR at end for reference
    std::cout << std::left  << std::setw(28) << "QR (full, reference)"
              << std::right << std::setw(10) << std::fixed
                            << std::setprecision(1) << Scalar(prob.m) / prob.n
              << std::setw(10) << (qr_result.converged ? "YES" : "NO")
              << std::setw(8)  << qr_result.iters_to_converge
              << std::setw(16) << std::scientific << std::setprecision(4)
                               << qr_result.final_residual
              << std::setw(16) << std::fixed << std::setprecision(1)
                               << qr_result.total_time_ms
              << std::setw(10) << std::setprecision(2) << 1.0 << "x"
              << "\n";
    std::cout << std::string(100, '=') << "\n";
}

// ============================================================
// Per-iteration timing breakdown
// ============================================================
void print_timing_breakdown(const std::vector<RunResult>& results) {
    std::cout << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  TIMING BREAKDOWN (averages per iteration)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::left  << std::setw(32) << "Method"
              << std::right << std::setw(14) << "Avg solve(ms)"
              << std::setw(14) << "Avg jac(ms)"
              << std::setw(10) << "Solve %"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    for (const auto& r : results) {
        int n = static_cast<int>(r.iters.size());
        if (n == 0) continue;
        double avg_solve = r.total_solve_ms / n;
        double avg_jac   = r.total_jacobian_ms / n;
        double solve_pct = 100.0 * r.total_solve_ms
                           / (r.total_solve_ms + r.total_jacobian_ms + 1e-9);
        std::cout << std::left  << std::setw(32) << r.label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(14) << avg_solve
                  << std::setw(14) << avg_jac
                  << std::setw(10) << solve_pct << "%"
                  << "\n";
    }
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    // --- Parse arguments ---
    int    m        = (argc > 1) ? std::stoi(argv[1]) : 50000;
    int    n        = (argc > 2) ? std::stoi(argv[2]) : 100;
    int    max_iter = (argc > 3) ? std::stoi(argv[3]) : 50;
    Scalar tol      = (argc > 4) ? std::stod(argv[4]) : 1e-4;

    std::cout << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Gauss-Newton Experiment: QR vs Sketched Least Squares\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Problem : min_x || sin(A x) - b ||^2\n";
    std::cout << "  m       : " << m        << "  (residual dimension)\n";
    std::cout << "  n       : " << n        << "  (parameter dimension)\n";
    std::cout << "  max_iter: " << max_iter << "\n";
    std::cout << "  tol     : " << std::scientific << tol << "\n";
    std::cout << std::string(70, '=') << "\n\n";

    // --- Generate problem ---
    std::cout << "[1/4] Generating problem...\n";
    Problem prob = make_problem(m, n, 42);

    // --- Initial point: small random perturbation of zero ---
    Vector x0 = 0.1 * Vector::Random(n);

    // --- Define all solver configurations ---
    int k2n  = std::min(2  * n, m);
    int k4n  = std::min(4  * n, m);
    int k6n  = std::min(6  * n, m);
    int k8n  = std::min(8  * n, m);

    std::vector<std::pair<std::string, LinearSolver>> configs = {
        {"QR (HouseholderQR)",            make_qr_solver()},
        {"NormalEq (LDLT)",               make_normal_solver()},
        {"GaussSketch k=2n",              make_gaussian_sketch_solver(k2n)},
        {"GaussSketch k=4n",              make_gaussian_sketch_solver(k4n)},
        {"GaussSketch k=6n",              make_gaussian_sketch_solver(k6n)},
        {"GaussSketch k=8n",              make_gaussian_sketch_solver(k8n)},
        {"SparseSketch(s=1) k=4n",        make_sparse_sketch_solver(k4n, 1)},
        {"SparseSketch(s=4) k=4n",        make_sparse_sketch_solver(k4n, 4)},
    };

    // --- Run all configurations ---
    std::cout << "[2/4] Running Gauss-Newton for " << configs.size()
              << " solver configurations...\n\n";

    std::vector<RunResult> results;
    results.reserve(configs.size());

    for (auto& [label, solver] : configs) {
        std::cout << "  Running: " << label << "\n";
        auto res = run_gauss_newton(prob, x0, solver, label, max_iter, tol, false);
        results.push_back(res);
        std::cout << "    → " << (res.converged ? "CONVERGED" : "NOT CONVERGED")
                  << " in " << res.iters_to_converge << " iters"
                  << ",  ||r||_final = " << std::scientific << std::setprecision(3)
                  << res.final_residual
                  << ",  total = " << std::fixed << std::setprecision(1)
                  << res.total_time_ms << " ms\n";
    }

    // --- Print detailed residual trace for key methods ---
    std::cout << "\n[3/4] Residual decay traces...\n";
    std::vector<std::string> trace_labels = {
        "QR (HouseholderQR)",
        "GaussSketch k=4n",
        "GaussSketch k=2n",
        "SparseSketch(s=1) k=4n",
    };
    for (const auto& r : results) {
        for (const auto& lbl : trace_labels) {
            if (r.label == lbl) {
                print_residual_trace(r);
                break;
            }
        }
    }

    // --- Summary table ---
    std::cout << "\n[4/4] Summary...\n";
    print_summary_table(results);
    print_timing_breakdown(results);

    // --- Sketch size sensitivity ---
    std::cout << "\n[Bonus] Running sketch-size sensitivity study...\n";
    sketch_size_study(prob, x0, max_iter, tol);

    // --- Export CSV ---
    export_csv(results, "gauss_newton_results.csv");

    std::cout << "\nDone. To plot results:\n";
    std::cout << "  python3 -c \"\n";
    std::cout << "  import pandas as pd, matplotlib.pyplot as plt\n";
    std::cout << "  df = pd.read_csv('gauss_newton_results.csv')\n";
    std::cout << "  for m, g in df.groupby('method'):\n";
    std::cout << "      plt.semilogy(g['iter'], g['residual_norm'], label=m)\n";
    std::cout << "  plt.legend(); plt.xlabel('Iteration'); plt.ylabel('||r||')\n";
    std::cout << "  plt.title('Gauss-Newton Residual Decay'); plt.savefig('residual.png')\n";
    std::cout << "  \"\n\n";

    return 0;
}