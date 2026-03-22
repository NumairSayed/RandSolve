#include "core/Precision.hpp"
#include "rng/StdRNG.hpp"
#include "sketching/GaussianSketch.hpp"
#include "sketching/SparseSketch.hpp"
#include "solvers/QRSolver.hpp"
#include "randomized/SketchedLeastSquares.hpp"
#include "lowrank/RandomizedSVD.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <vector>

using Scalar = double;
using Traits = randnla::PrecisionTraits<Scalar>;
using Matrix = Traits::Matrix;
using Vector = Traits::Vector;

using Clock = std::chrono::high_resolution_clock;

double time_ms(auto start, auto end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

//////////////////////////////////////////////////////////////////
// 1. Linear Regression Benchmark
//////////////////////////////////////////////////////////////////
void benchmark_regression() {
    std::cout << "\n=== Linear Regression ===\n";

    int m = 2000, n = 50;
    Matrix A = Matrix::Random(m, n);
    Vector x_true = Vector::Random(n);
    Vector b = A * x_true;

    // Exact
    auto t1 = Clock::now();
    randnla::QRSolver<Scalar> solver;
    Vector x_exact = solver.solve(A, b);
    auto t2 = Clock::now();

    std::cout << "Exact: " << time_ms(t1, t2) << " ms\n";

    std::vector<int> ks = {2*n, 4*n, 8*n};

    for (int k : ks) {
        randnla::StdRNG<Scalar> rng(42);

        // Gaussian
        auto gsketch = std::make_shared<randnla::GaussianSketch<Scalar>>(k, m, rng);
        auto solver_ptr = std::make_shared<randnla::QRSolver<Scalar>>();
        randnla::SketchedLeastSquares<Scalar> sls(gsketch, solver_ptr);

        auto t3 = Clock::now();
        Vector x_hat = sls.solve(A, b);
        auto t4 = Clock::now();

        double err = (x_hat - x_exact).norm() / x_exact.norm();

        std::cout << "[Gaussian k=" << k << "] time="
                  << time_ms(t3, t4)
                  << " ms  rel_err=" << err << "\n";

        // Sparse
        auto ssketch = std::make_shared<randnla::SparseSketch<Scalar>>(k, m, rng, 1);
        randnla::SketchedLeastSquares<Scalar> sls2(ssketch, solver_ptr);

        auto t5 = Clock::now();
        Vector x_hat2 = sls2.solve(A, b);
        auto t6 = Clock::now();

        double err2 = (x_hat2 - x_exact).norm() / x_exact.norm();

        std::cout << "[Sparse   k=" << k << "] time="
                  << time_ms(t5, t6)
                  << " ms  rel_err=" << err2 << "\n";
    }
}

//////////////////////////////////////////////////////////////////
// 2. PCA Benchmark
//////////////////////////////////////////////////////////////////
void benchmark_pca() {
    std::cout << "\n=== PCA (Randomized SVD) ===\n";

    int m = 1000, n = 200, r = 10;
    Matrix A = Matrix::Random(m, r) * Matrix::Random(r, n);

    // Exact SVD
    auto t1 = Clock::now();
    Eigen::BDCSVD<Matrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Matrix U_exact = svd.matrixU().leftCols(r);
    auto t2 = Clock::now();

    std::cout << "Exact SVD: " << time_ms(t1, t2) << " ms\n";

    randnla::StdRNG<Scalar> rng(42);

    auto t3 = Clock::now();
    randnla::RandomizedSVD<Scalar> rsvd(20, 2);
    rsvd.compute(A, r, rng);
    auto t4 = Clock::now();

    Matrix U_approx = rsvd.matrixU().leftCols(r);

    double err = (U_exact * U_exact.transpose()
                 - U_approx * U_approx.transpose()).norm();

    std::cout << "RandSVD: " << time_ms(t3, t4)
              << " ms  subspace_err=" << err << "\n";
}

//////////////////////////////////////////////////////////////////
// 3. Attention Benchmark
//////////////////////////////////////////////////////////////////
void benchmark_attention() {
    std::cout << "\n=== Attention (QK^T Approximation) ===\n";

    int n = 512, d = 64;
    Matrix Q = Matrix::Random(n, d);
    Matrix K = Matrix::Random(n, d);

    // Exact
    auto t1 = Clock::now();
    Matrix exact = Q * K.transpose();
    auto t2 = Clock::now();

    std::cout << "Exact: " << time_ms(t1, t2) << " ms\n";

    std::vector<int> ks = {128,256,384};

    for (int k : ks) {
        randnla::StdRNG<Scalar> rng(42);

        // Gaussian
        randnla::GaussianSketch<Scalar> gsketch(k, n, rng);

        auto t3 = Clock::now();
        Matrix SQ = gsketch.apply(Q);
        Matrix SK = gsketch.apply(K);
        Matrix approx = SQ * SK.transpose();
        auto t4 = Clock::now();

        double err = (exact - approx).norm() / exact.norm();

        std::cout << "[Gaussian k=" << k << "] time="
                  << time_ms(t3, t4)
                  << " ms  rel_err=" << err << "\n";

        // Sparse
        randnla::SparseSketch<Scalar> ssketch(k, n, rng, 1);

        auto t5 = Clock::now();
        Matrix SQ2 = ssketch.apply(Q);
        Matrix SK2 = ssketch.apply(K);
        Matrix approx2 = SQ2 * SK2.transpose();
        auto t6 = Clock::now();

        double err2 = (exact - approx2).norm() / exact.norm();

        std::cout << "[Sparse   k=" << k << "] time="
                  << time_ms(t5, t6)
                  << " ms  rel_err=" << err2 << "\n";
    }
}

//////////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////////
int main() {
    std::cout << "===== RandNLA Benchmark Suite =====\n";

    benchmark_regression();
    benchmark_pca();
    benchmark_attention();

    std::cout << "\nDone.\n";
    return 0;
}