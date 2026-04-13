/**
 * @file quality_verify.cpp
 * @brief Verify that forward_seq_sketch achieves cos-sim > 0.90
 *        before proceeding to PyTorch integration.
 *
 * This is a go/no-go gate. Do not proceed to pybind11 bindings
 * until this script prints PASS for all configurations.
 */

 #include "attention/SketchedAttention.hpp"
 #include "core/Precision.hpp"
 #include <Eigen/Dense>
 #include <iostream>
 #include <iomanip>
 #include <vector>
 #include <chrono>
 #include <numeric>
 #include <cmath>
 
 using Scalar = double;
 using Matrix = randnla::PrecisionTraits<Scalar>::Matrix;
 using Clock  = std::chrono::high_resolution_clock;
 using Ms     = std::chrono::duration<double, std::milli>;
 namespace attn = randnla::attention;
 
 // ── Metrics ────────────────────────────────────────────────────────────────
 
 Scalar cosine_sim(const Matrix& A, const Matrix& B) {
     Eigen::Map<const Eigen::VectorXd> a(A.data(), A.size());
     Eigen::Map<const Eigen::VectorXd> b(B.data(), B.size());
     Scalar n = a.norm() * b.norm();
     return (n > 1e-12) ? a.dot(b) / n : 0.0;
 }
 
 Scalar rel_err(const Matrix& A, const Matrix& B) {
     return (A - B).norm() / (A.norm() + 1e-12);
 }
 
 // Average over multiple random seeds to get stable estimate
 struct QualityStats {
     Scalar mean_cos, std_cos, min_cos, mean_err;
 };
 
 QualityStats measure_quality(int L, int d, int k_seq,
                               int trials = 15) {
     attn::ExactAttention<Scalar>::Config ecfg;
     attn::ExactAttention<Scalar> exact(ecfg);
 
     attn::SketchedAttention<Scalar>::Config scfg;
     scfg.sketch_dim      = d;
     scfg.rng_seed        = 0;
     scfg.reseed_per_call = true;
     attn::SketchedAttention<Scalar> sk(scfg);
 
     std::vector<Scalar> cos_vals, err_vals;
 
     for (int t = 0; t < trials; ++t) {
         // Fresh random Q, K, V each trial
         srand(t * 97 + 13);
         Matrix Q = Matrix::Random(L, d);
         Matrix K = Matrix::Random(L, d);
         Matrix V = Matrix::Random(L, d);
 
         // Normalise rows of Q and K (mimics real attention where
         // projections are layer-normed)
         for (int i = 0; i < L; ++i) {
             Q.row(i) /= (Q.row(i).norm() + 1e-8);
             K.row(i) /= (K.row(i).norm() + 1e-8);
         }
 
         auto exact_res = exact.forward(Q, K, V);
         auto sk_res    = sk.forward_seq_sketch(Q, K, V, k_seq);
 
         cos_vals.push_back(cosine_sim(exact_res.output, sk_res.output));
         err_vals.push_back(rel_err(exact_res.output,    sk_res.output));
     }
 
     Scalar mean_c = std::accumulate(cos_vals.begin(), cos_vals.end(), 0.0) / trials;
     Scalar min_c  = *std::min_element(cos_vals.begin(), cos_vals.end());
     Scalar var_c  = 0.0;
     for (auto c : cos_vals) var_c += (c - mean_c) * (c - mean_c);
     Scalar std_c  = std::sqrt(var_c / (trials - 1));
     Scalar mean_e = std::accumulate(err_vals.begin(), err_vals.end(), 0.0) / trials;
 
     return {mean_c, std_c, min_c, mean_e};
 }
 
 // ── Timing ─────────────────────────────────────────────────────────────────
 
 double measure_speedup(int L, int d, int k_seq, int runs = 5) {
     srand(42);
     Matrix Q = Matrix::Random(L, d);
     Matrix K = Matrix::Random(L, d);
     Matrix V = Matrix::Random(L, d);
 
     attn::ExactAttention<Scalar>::Config ecfg;
     attn::ExactAttention<Scalar> exact(ecfg);
 
     attn::SketchedAttention<Scalar>::Config scfg;
     scfg.sketch_dim = d;
     scfg.rng_seed   = 42;
     attn::SketchedAttention<Scalar> sk(scfg);
 
     // Warmup
     exact.forward(Q, K, V);
     sk.forward_seq_sketch(Q, K, V, k_seq);
 
     double t_exact = 1e18, t_sk = 1e18;
     for (int r = 0; r < runs; ++r) {
         auto t0 = Clock::now();
         exact.forward(Q, K, V);
         t_exact = std::min(t_exact, Ms(Clock::now() - t0).count());
 
         t0 = Clock::now();
         sk.forward_seq_sketch(Q, K, V, k_seq);
         t_sk = std::min(t_sk, Ms(Clock::now() - t0).count());
     }
     return t_exact / t_sk;
 }
 
 // ── Main ───────────────────────────────────────────────────────────────────
 
 int main() {
     std::cout << "\n";
     std::cout << std::string(85, '=') << "\n";
     std::cout << "  QUALITY VERIFICATION — Row-Sampled Landmark Attention\n";
     std::cout << "  GO/NO-GO gate before PyTorch integration\n";
     std::cout << std::string(85, '=') << "\n\n";
 
     // ── Table 1: Quality vs k/L ratio ──────────────────────────────────
     std::cout << "  [1/3] Quality vs sketch fraction  (d=64, 15 trials each)\n\n";
 
     std::cout << std::left  << std::setw(8)  << "L"
               << std::setw(8)  << "k"
               << std::setw(8)  << "k/L"
               << std::right
               << std::setw(14) << "Mean CosSim"
               << std::setw(12) << "Std"
               << std::setw(12) << "Min"
               << std::setw(12) << "RelErr"
               << std::setw(10) << "Gate"
               << "\n";
     std::cout << std::string(85, '-') << "\n";
 
     int d = 64;
     bool all_pass = true;
 
     std::vector<std::pair<int,int>> configs = {
         {512,  512/4},  {512,  512/8},  {512,  512/16},
         {1024, 1024/4}, {1024, 1024/8}, {1024, 1024/16},
         {2048, 2048/4}, {2048, 2048/8}, {2048, 2048/16},
         {4096, 4096/4}, {4096, 4096/8}, {4096, 4096/16},
     };
 
     for (auto [L, k] : configs) {
         auto s = measure_quality(L, d, k);
         bool pass = (s.mean_cos > 0.90) && (s.min_cos > 0.80);
         if (!pass) all_pass = false;
 
         std::cout << std::left  << std::setw(8)  << L
                   << std::setw(8)  << k
                   << std::setw(8)  << std::fixed << std::setprecision(3)
                                    << Scalar(k)/L
                   << std::right
                   << std::setw(14) << std::setprecision(6) << s.mean_cos
                   << std::setw(12) << s.std_cos
                   << std::setw(12) << s.min_cos
                   << std::setw(12) << s.mean_err
                   << std::setw(10) << (pass ? "✓ PASS" : "✗ FAIL")
                   << "\n";
     }
 
     // ── Table 2: Speedup ───────────────────────────────────────────────
     std::cout << "\n  [2/3] Speedup vs sequence length  (d=64, k=L/8)\n\n";
 
     std::cout << std::left  << std::setw(8)  << "L"
               << std::setw(8)  << "k"
               << std::right
               << std::setw(14) << "Speedup"
               << std::setw(12) << "Gate (>1x)"
               << "\n";
     std::cout << std::string(44, '-') << "\n";
 
     for (int L : {512, 1024, 2048, 4096, 8192}) {
         int k        = L / 8;
         double sp    = measure_speedup(L, d, k);
         bool sp_pass = sp > 1.0;
         if (!sp_pass) all_pass = false;
         std::cout << std::left  << std::setw(8)  << L
                   << std::setw(8)  << k
                   << std::right << std::fixed << std::setprecision(2)
                   << std::setw(14) << sp << "x"
                   << std::setw(12) << (sp_pass ? "✓ PASS" : "✗ FAIL")
                   << "\n";
     }
 
     // ── Table 3: Normalised vs unnormalised Q/K ────────────────────────
     std::cout << "\n  [3/3] Effect of row-normalising Q, K  (L=1024, d=64, k=128)\n\n";
 
     std::cout << std::left  << std::setw(20) << "Input type"
               << std::right
               << std::setw(14) << "Mean CosSim"
               << std::setw(12) << "Min CosSim"
               << "\n";
     std::cout << std::string(48, '-') << "\n";
 
     // Helper: measure without normalisation
     auto measure_raw = [&](bool normalise) -> std::pair<Scalar,Scalar> {
         attn::ExactAttention<Scalar>::Config ecfg;
         attn::ExactAttention<Scalar> exact(ecfg);
         attn::SketchedAttention<Scalar>::Config scfg;
         scfg.rng_seed = 0; scfg.reseed_per_call = true;
         attn::SketchedAttention<Scalar> sk(scfg);
 
         std::vector<Scalar> vals;
         for (int t = 0; t < 15; ++t) {
             srand(t * 13 + 7);
             Matrix Q = Matrix::Random(1024, 64);
             Matrix K = Matrix::Random(1024, 64);
             Matrix V = Matrix::Random(1024, 64);
             if (normalise) {
                 for (int i = 0; i < 1024; ++i) {
                     Q.row(i) /= (Q.row(i).norm() + 1e-8);
                     K.row(i) /= (K.row(i).norm() + 1e-8);
                 }
             }
             auto er = exact.forward(Q, K, V);
             auto sr = sk.forward_seq_sketch(Q, K, V, 128);
             vals.push_back(cosine_sim(er.output, sr.output));
         }
         Scalar m = std::accumulate(vals.begin(), vals.end(), 0.0) / 15;
         Scalar mn = *std::min_element(vals.begin(), vals.end());
         return {m, mn};
     };
 
     auto [m_raw, mn_raw]  = measure_raw(false);
     auto [m_norm, mn_norm] = measure_raw(true);
     std::cout << std::left  << std::setw(20) << "Raw (no norm)"
               << std::right << std::fixed << std::setprecision(6)
               << std::setw(14) << m_raw
               << std::setw(12) << mn_raw << "\n";
     std::cout << std::left  << std::setw(20) << "Row-normalised"
               << std::right
               << std::setw(14) << m_norm
               << std::setw(12) << mn_norm << "\n";
 
     // ── Verdict ────────────────────────────────────────────────────────
     std::cout << "\n" << std::string(85, '=') << "\n";
     if (all_pass) {
         std::cout << "  VERDICT: ✓ ALL PASS — proceed to PyTorch integration\n";
     } else {
         std::cout << "  VERDICT: ✗ FAILURES DETECTED — fix quality before proceeding\n";
         std::cout << "  Check: is forward_seq_sketch using the same idx for K and V?\n";
         std::cout << "  Check: are includes <numeric> <random> <algorithm> present?\n";
     }
     std::cout << std::string(85, '=') << "\n\n";
 
     return all_pass ? 0 : 1;
 }