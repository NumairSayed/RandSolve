/**
 * @file attention_benchmark.cpp
 * @brief Research benchmark: Exact vs Sketched Self-Attention.
 *
 * Experiments:
 *   1. Scaling study       — time vs sequence length L ∈ {128..8192}
 *   2. Sketch size ablation— quality vs k ∈ {d/8 .. d}
 *   3. Sketch type study   — Gaussian vs CountSketch vs OSNAP
 *   4. Gradient fidelity   — cosine similarity of output matrices
 *   5. Multi-head scaling  — heads ∈ {1, 2, 4, 8}
 *   6. Attention entropy   — distribution sharpness comparison
 *   7. Sequence sketch     — speedup vs exact for k_seq ≪ L
 *
 * All results exported to CSV for plotting.
 *
 * Usage:
 *   ./attention_benchmark [d_model] [max_seq_len] [num_heads]
 *   ./attention_benchmark 64 4096 1         # single head, d=64
 *   ./attention_benchmark 512 2048 8        # multi-head
 */

 #include "attention/SketchedAttention.hpp"
 #include "core/Precision.hpp"
 
 #include <Eigen/Dense>
 #include <iostream>
 #include <iomanip>
 #include <fstream>
 #include <vector>
 #include <chrono>
 #include <cmath>
 #include <string>
 #include <numeric>
 #include <algorithm>
 #include <sstream>
 
 using Scalar  = double;
 using Matrix  = randnla::PrecisionTraits<Scalar>::Matrix;
 using Vector  = randnla::PrecisionTraits<Scalar>::Vector;
 using Clock   = std::chrono::high_resolution_clock;
 using Ms      = std::chrono::duration<double, std::milli>;
 
 namespace attn = randnla::attention;
 
 inline bool srht_dim_ok(int d) { return d > 0 && (d & (d - 1)) == 0; }
 
 // ============================================================
 // Utilities
 // ============================================================
 
 // Run a lambda `runs` times, return minimum elapsed ms
 template <typename Fn>
 double min_time_ms(Fn&& fn, int runs = 5) {
     double best = 1e18;
     for (int r = 0; r < runs; ++r) {
         auto t0 = Clock::now();
         fn();
         double t = Ms(Clock::now() - t0).count();
         if (t < best) best = t;
     }
     return best;
 }
 
 // Cosine similarity between two matrices (flattened)
 Scalar cosine_similarity(const Matrix& A, const Matrix& B) {
     Vector a = Eigen::Map<const Vector>(A.data(), A.size());
     Vector b = Eigen::Map<const Vector>(B.data(), B.size());
     Scalar dot  = a.dot(b);
     Scalar norm = a.norm() * b.norm();
     return (norm > Scalar{1e-12}) ? dot / norm : Scalar{0};
 }
 
 // Relative Frobenius error ||A - B||_F / ||A||_F
 Scalar rel_frob_error(const Matrix& A, const Matrix& B) {
     Scalar denom = A.norm();
     return (denom > Scalar{1e-12}) ? (A - B).norm() / denom : (A - B).norm();
 }
 
 // Generate random Q, K, V for a given L, d
 void gen_qkv(int L, int d, Matrix& Q, Matrix& K, Matrix& V, uint64_t seed = 0) {
     srand(static_cast<unsigned>(seed));
     Q = Matrix::Random(L, d);
     K = Matrix::Random(L, d);
     V = Matrix::Random(L, d);
 }
 
 void print_header(const std::string& title) {
     std::cout << "\n" << std::string(90, '=') << "\n";
     std::cout << "  " << title << "\n";
     std::cout << std::string(90, '=') << "\n";
 }
 
 // ============================================================
 // CSV writer
 // ============================================================
 struct CsvWriter {
     std::ofstream f;
     explicit CsvWriter(const std::string& path) : f(path) {
         if (!f.is_open())
             std::cerr << "[WARN] Cannot open " << path << "\n";
     }
     template <typename... Args>
     void row(Args&&... args) {
         bool first = true;
         auto write_one = [&](const auto& v) {
             if (!first) f << ",";
             f << v;
             first = false;
         };
         (write_one(args), ...);
         f << "\n";
     }
 };
 
 // ============================================================
 // Experiment 1: Scaling — time vs sequence length
 // ============================================================
 void exp1_scaling(int d, int num_heads, CsvWriter& csv) {
     print_header("EXPERIMENT 1: Time vs Sequence Length  (d=" +
                  std::to_string(d) + ", heads=" + std::to_string(num_heads) + ")");
 
     std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096};
     // Only go to 8192 if d is small enough to be tractable
     if (d <= 64) seq_lens.push_back(8192);
 
     int k = d;  // sketch dim = d (square)
 
     csv.row("exp", "L", "d", "k", "method", "time_ms", "speedup");
 
     // Table header
     std::cout << std::left  << std::setw(8)  << "L"
               << std::right << std::setw(14) << "Exact (ms)"
               << std::setw(16) << "Gauss k=d"
               << std::setw(16) << "Sparse k=d"
               << std::setw(14) << "SRHT k=d"
               << std::setw(10) << "Sp(G)"
               << std::setw(10) << "Sp(S)"
               << std::setw(10) << "Sp(H)"
               << "\n";
     std::cout << std::string(100, '-') << "\n";
 
     for (int L : seq_lens) {
         Matrix Q, K, V;
         gen_qkv(L, d, Q, K, V, 42);
 
         // Exact
         attn::ExactAttention<Scalar>::Config ecfg;
         attn::ExactAttention<Scalar> exact(ecfg);
         double t_exact = min_time_ms([&]{ exact.forward(Q, K, V); });
 
         // Gaussian sketch
         attn::SketchedAttention<Scalar>::Config gcfg;
         gcfg.sketch_dim  = k;
         gcfg.sketch_type = attn::SketchType::GAUSSIAN;
         attn::SketchedAttention<Scalar> gauss(gcfg);
         double t_gauss = min_time_ms([&]{ gauss.forward(Q, K, V); });
 
         // Sparse sketch (CountSketch)
         attn::SketchedAttention<Scalar>::Config scfg;
         scfg.sketch_dim  = k;
         scfg.sketch_type = attn::SketchType::SPARSE_COUNT;
         attn::SketchedAttention<Scalar> sparse(scfg);
         double t_sparse = min_time_ms([&]{ sparse.forward(Q, K, V); });
 
         double t_srht = -1.0;
         double sp_h   = 0.0;
         if (srht_dim_ok(d)) {
             attn::SketchedAttention<Scalar>::Config hcfg;
             hcfg.sketch_dim  = k;
             hcfg.sketch_type = attn::SketchType::SRHT;
             attn::SketchedAttention<Scalar> srht_sk(hcfg);
             t_srht = min_time_ms([&]{ srht_sk.forward(Q, K, V); });
             sp_h   = t_exact / t_srht;
         }
 
         double sp_g = t_exact / t_gauss;
         double sp_s = t_exact / t_sparse;
 
         std::cout << std::left  << std::setw(8)  << L
                   << std::right << std::fixed << std::setprecision(2)
                   << std::setw(14) << t_exact
                   << std::setw(16) << t_gauss
                   << std::setw(16) << t_sparse;
         if (srht_dim_ok(d))
             std::cout << std::setw(14) << t_srht
                       << std::setw(10) << sp_g
                       << std::setw(10) << sp_s
                       << std::setw(10) << sp_h << "\n";
         else
             std::cout << std::setw(14) << "n/a"
                       << std::setw(10) << sp_g
                       << std::setw(10) << sp_s
                       << std::setw(10) << "n/a" << "\n";
 
         csv.row("scaling", L, d, k, "exact",        t_exact,  1.0);
         csv.row("scaling", L, d, k, "gauss_sketch",  t_gauss,  sp_g);
         csv.row("scaling", L, d, k, "sparse_sketch", t_sparse, sp_s);
         if (srht_dim_ok(d))
             csv.row("scaling", L, d, k, "srht_sketch", t_srht, sp_h);
     }
 }
 
 // ============================================================
 // Experiment 2: Sketch size ablation — quality vs k
 // ============================================================
 void exp2_sketch_size(int L, int d, CsvWriter& csv) {
     print_header("EXPERIMENT 2: Quality vs Sketch Dimension k  (L=" +
                  std::to_string(L) + ", d=" + std::to_string(d) + ")");
 
     Matrix Q, K, V;
     gen_qkv(L, d, Q, K, V, 7);
 
     // Ground truth
     attn::ExactAttention<Scalar>::Config ecfg;
     ecfg.compute_diag = true;
     attn::ExactAttention<Scalar> exact(ecfg);
     auto exact_res = exact.forward(Q, K, V);
 
     csv.row("exp", "L", "d", "k", "k_over_d", "method",
             "output_cos_sim", "output_rel_err",
             "attn_cos_sim", "attn_rel_err",
             "time_ms", "speedup");
 
     std::cout << std::left  << std::setw(12) << "Method"
               << std::setw(8)  << "k"
               << std::setw(8)  << "k/d"
               << std::right
               << std::setw(14) << "Out CosSim"
               << std::setw(14) << "Out RelErr"
               << std::setw(14) << "Attn CosSim"
               << std::setw(12) << "Time(ms)"
               << std::setw(10) << "Speedup"
               << "\n";
     std::cout << std::string(96, '-') << "\n";
 
     // Print exact as reference
     double t_exact = min_time_ms([&]{ exact.forward(Q, K, V); });
     std::cout << std::left  << std::setw(12) << "EXACT"
               << std::setw(8)  << "-"
               << std::setw(8)  << "-"
               << std::right
               << std::setw(14) << "1.000000"
               << std::setw(14) << "0.000000"
               << std::setw(14) << "1.000000"
               << std::setw(12) << std::fixed << std::setprecision(2) << t_exact
               << std::setw(10) << "1.00x"
               << "\n";
 
     std::vector<int> k_vals;
     for (int f : {1, 2, 4, 8, 16, 32})
         if (d / f >= 1) k_vals.push_back(std::max(1, d / f));
     k_vals.push_back(d);
     if (d * 2 <= L) k_vals.push_back(d * 2);
     std::sort(k_vals.begin(), k_vals.end());
     k_vals.erase(std::unique(k_vals.begin(), k_vals.end()), k_vals.end());
 
     struct SketchMethod {
         const char*           name;
         attn::SketchType      type;
     };
     std::vector<SketchMethod> methods = {
         {"gaussian", attn::SketchType::GAUSSIAN},
     };
     if (srht_dim_ok(d))
         methods.push_back({"srht", attn::SketchType::SRHT});
 
     for (int k : k_vals) {
         for (const auto& sm : methods) {
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.sketch_dim   = k;
             cfg.sketch_type  = sm.type;
             cfg.compute_diag = true;
             attn::SketchedAttention<Scalar> sketched(cfg);
 
             auto sk_res = sketched.forward(Q, K, V);
 
             Scalar out_cos = cosine_similarity(exact_res.output, sk_res.output);
             Scalar out_err = rel_frob_error(exact_res.output, sk_res.output);
             Scalar att_cos =
                 cosine_similarity(exact_res.attention_weights, sk_res.attention_weights);
 
             double t_sk    = min_time_ms([&]{ sketched.forward(Q, K, V); });
             double speedup = t_exact / t_sk;
 
             std::cout << std::left  << std::setw(12) << sm.name
                       << std::setw(8)  << k
                       << std::setw(8)  << std::fixed << std::setprecision(2)
                                        << Scalar(k) / d
                       << std::right
                       << std::setw(14) << std::setprecision(6) << out_cos
                       << std::setw(14) << out_err
                       << std::setw(14) << att_cos
                       << std::setw(12) << std::setprecision(2) << t_sk
                       << std::setw(10) << speedup << "x"
                       << "\n";
 
             csv.row("ablation", L, d, k, Scalar(k) / d, sm.name, out_cos, out_err,
                     att_cos, att_cos, t_sk, speedup);
         }
     }
 }
 
 // ============================================================
 // Experiment 3: Sketch type comparison
 // ============================================================
 void exp3_sketch_types(int L, int d, CsvWriter& csv) {
     print_header("EXPERIMENT 3: Sketch Type Comparison  (L=" +
                  std::to_string(L) + ", d=" + std::to_string(d) + ")");
 
     Matrix Q, K, V;
     gen_qkv(L, d, Q, K, V, 13);
 
     attn::ExactAttention<Scalar>::Config ecfg;
     attn::ExactAttention<Scalar> exact(ecfg);
     auto exact_res = exact.forward(Q, K, V);
     double t_exact = min_time_ms([&]{ exact.forward(Q, K, V); });
 
     csv.row("exp", "L", "d", "k", "sketch_type",
             "output_cos_sim", "output_rel_err", "time_ms", "speedup");
 
     std::cout << std::left  << std::setw(24) << "Sketch Type"
               << std::setw(6)  << "k"
               << std::right
               << std::setw(16) << "CosSim Output"
               << std::setw(16) << "RelErr Output"
               << std::setw(14) << "Time(ms)"
               << std::setw(10) << "Speedup"
               << "\n";
     std::cout << std::string(90, '-') << "\n";
 
     // Print exact
     std::cout << std::left  << std::setw(24) << "Exact (baseline)"
               << std::setw(6)  << d
               << std::right
               << std::setw(16) << "1.000000"
               << std::setw(16) << "0.000000"
               << std::setw(14) << std::fixed << std::setprecision(2) << t_exact
               << std::setw(10) << "1.00x"
               << "\n";
 
     struct TypeConfig {
         std::string name;
         attn::SketchType type;
         int sparse_s;
     };
 
     std::vector<TypeConfig> types = {
         {"Gaussian",        attn::SketchType::GAUSSIAN,      1},
         {"SRHT",            attn::SketchType::SRHT,          1},
         {"CountSketch s=1", attn::SketchType::SPARSE_COUNT,  1},
         {"OSNAP s=2",       attn::SketchType::SPARSE_OSNAP,  2},
         {"OSNAP s=4",       attn::SketchType::SPARSE_OSNAP,  4},
     };
 
     for (int k : {d/2, d}) {
         for (auto& tc : types) {
             if (tc.type == attn::SketchType::SRHT && !srht_dim_ok(d))
                 continue;
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.sketch_dim  = k;
             cfg.sketch_type = tc.type;
             cfg.sparse_s    = tc.sparse_s;
             attn::SketchedAttention<Scalar> sk(cfg);
 
             auto sk_res = sk.forward(Q, K, V);
             Scalar cos_sim = cosine_similarity(exact_res.output, sk_res.output);
             Scalar rel_err = rel_frob_error(exact_res.output, sk_res.output);
             double t_sk    = min_time_ms([&]{ sk.forward(Q, K, V); });
             double speedup = t_exact / t_sk;
 
             std::string label = tc.name + " k=" + std::to_string(k);
             std::cout << std::left  << std::setw(24) << label
                       << std::setw(6)  << k
                       << std::right
                       << std::setw(16) << std::fixed << std::setprecision(6) << cos_sim
                       << std::setw(16) << rel_err
                       << std::setw(14) << std::setprecision(2) << t_sk
                       << std::setw(10) << speedup << "x"
                       << "\n";
 
             csv.row("sketch_type", L, d, k, label, cos_sim, rel_err, t_sk, speedup);
         }
     }
 }
 
 // ============================================================
 // Experiment 4: Gradient fidelity
 // (proxy: cosine similarity of output O across many random inputs)
 // ============================================================
 void exp4_gradient_fidelity(int L, int d, CsvWriter& csv) {
     print_header("EXPERIMENT 4: Output Fidelity Across Random Inputs  (L=" +
                  std::to_string(L) + ", d=" + std::to_string(d) + ")");
 
     std::cout << "  Measuring cosine similarity of O_sketched vs O_exact\n";
     std::cout << "  across 20 independent random (Q,K,V) draws.\n\n";
 
     int trials = 20;
     std::vector<int> k_vals = {d/4, d/2, d, 2*d};
     for (int& kv : k_vals) kv = std::max(1, std::min(kv, d));
     std::sort(k_vals.begin(), k_vals.end());
     k_vals.erase(std::unique(k_vals.begin(), k_vals.end()), k_vals.end());
 
     attn::ExactAttention<Scalar>::Config ecfg;
     attn::ExactAttention<Scalar> exact(ecfg);
 
     csv.row("exp", "trial", "k", "method", "cos_sim", "rel_err");
 
     struct FidM {
         const char*      name;
         attn::SketchType type;
     };
     std::vector<FidM> fid_methods = {{"gaussian", attn::SketchType::GAUSSIAN}};
     if (srht_dim_ok(d))
         fid_methods.push_back({"srht", attn::SketchType::SRHT});
 
     for (const auto& fm : fid_methods) {
         std::cout << "  --- " << fm.name << " ---\n";
         std::cout << std::left  << std::setw(8)  << "k"
                   << std::right << std::setw(14) << "Mean CosSim"
                   << std::setw(14) << "Std CosSim"
                   << std::setw(14) << "Min CosSim"
                   << std::setw(14) << "Mean RelErr"
                   << "\n";
         std::cout << std::string(60, '-') << "\n";
 
         for (int k : k_vals) {
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.sketch_dim      = k;
             cfg.sketch_type     = fm.type;
             cfg.reseed_per_call = true;
             attn::SketchedAttention<Scalar> sk(cfg);
 
             std::vector<Scalar> cos_sims, rel_errs;
             for (int t = 0; t < trials; ++t) {
                 Matrix Q, K, V;
                 gen_qkv(L, d, Q, K, V, static_cast<uint64_t>(t) * 17 + 3);
 
                 auto exact_res = exact.forward(Q, K, V);
                 auto sk_res    = sk.forward(Q, K, V);
 
                 cos_sims.push_back(cosine_similarity(exact_res.output, sk_res.output));
                 rel_errs.push_back(rel_frob_error(exact_res.output, sk_res.output));
 
                 csv.row("fidelity", t, k, fm.name, cos_sims.back(), rel_errs.back());
             }
 
             Scalar mean_cos = std::accumulate(cos_sims.begin(), cos_sims.end(), Scalar{0})
                               / trials;
             Scalar min_cos  = *std::min_element(cos_sims.begin(), cos_sims.end());
             Scalar var_cos  = Scalar{0};
             for (auto c : cos_sims) var_cos += (c - mean_cos) * (c - mean_cos);
             Scalar std_cos  = std::sqrt(var_cos / (trials - 1));
             Scalar mean_err = std::accumulate(rel_errs.begin(), rel_errs.end(), Scalar{0})
                               / trials;
 
             std::cout << std::left  << std::setw(8)  << k
                       << std::right << std::fixed << std::setprecision(6)
                       << std::setw(14) << mean_cos
                       << std::setw(14) << std_cos
                       << std::setw(14) << min_cos
                       << std::setw(14) << mean_err
                       << "\n";
         }
         std::cout << "\n";
     }
 }
 
 // ============================================================
 // Experiment 5: Multi-head scaling
 // ============================================================
 void exp5_multihead(int L, int d_model, CsvWriter& csv) {
     print_header("EXPERIMENT 5: Multi-Head Scaling  (L=" +
                  std::to_string(L) + ", d_model=" + std::to_string(d_model) + ")");
 
     csv.row("exp", "L", "d_model", "num_heads", "d_head",
             "sketch_dim", "method", "time_ms", "speedup");
 
     std::cout << std::left  << std::setw(10) << "Heads"
               << std::setw(8)  << "d_head"
               << std::setw(10) << "k"
               << std::setw(12) << "Sketch"
               << std::right
               << std::setw(14) << "Exact (ms)"
               << std::setw(16) << "Sketched (ms)"
               << std::setw(12) << "Speedup"
               << "\n";
     std::cout << std::string(88, '-') << "\n";
 
     for (int num_heads : {1, 2, 4, 8}) {
         if (d_model % num_heads != 0) continue;
         int d_head = d_model / num_heads;
         int k      = std::max(1, d_head / 2);
 
         Matrix Q = Matrix::Random(L, d_model);
         Matrix K = Matrix::Random(L, d_model);
         Matrix V = Matrix::Random(L, d_model);
 
         // Exact: run as independent single heads
         double t_exact = min_time_ms([&]() {
             for (int h = 0; h < num_heads; ++h) {
                 attn::ExactAttention<Scalar>::Config ec;
                 attn::ExactAttention<Scalar> e(ec);
                 Matrix Qh = Q.middleCols(h * d_head, d_head);
                 Matrix Kh = K.middleCols(h * d_head, d_head);
                 Matrix Vh = V.middleCols(h * d_head, d_head);
                 e.forward(Qh, Kh, Vh);
             }
         });
 
         csv.row("multihead", L, d_model, num_heads, d_head, k, "exact", t_exact, 1.0);

         struct MHSketch {
             const char*      name;
             attn::SketchType type;
         };
         std::vector<MHSketch> mh_sk = {
             {"gaussian", attn::SketchType::GAUSSIAN},
         };
         if (srht_dim_ok(d_head))
             mh_sk.push_back({"srht", attn::SketchType::SRHT});

         for (const auto& ms : mh_sk) {
             attn::MultiHeadSketchedAttention<Scalar>::Config mhcfg;
             mhcfg.num_heads   = num_heads;
             mhcfg.d_model     = d_model;
             mhcfg.sketch_dim  = k;
             mhcfg.sketch_type = ms.type;
             attn::MultiHeadSketchedAttention<Scalar> mh(mhcfg);

             double t_sk    = min_time_ms([&] { mh.forward(Q, K, V); });
             double speedup = t_exact / t_sk;

             std::cout << std::left  << std::setw(10) << num_heads
                       << std::setw(8)  << d_head
                       << std::setw(10) << k
                       << std::setw(12)  << ms.name
                       << std::right << std::fixed << std::setprecision(2)
                       << std::setw(14) << t_exact
                       << std::setw(16) << t_sk
                       << std::setw(12) << speedup << "x"
                       << "\n";

             csv.row("multihead", L, d_model, num_heads, d_head, k, ms.name, t_sk,
                     speedup);
         }
     }
 }
 
 // ============================================================
 // Experiment 6: Attention entropy comparison
 // ============================================================
 void exp6_entropy(int L, int d, CsvWriter& csv) {
     print_header("EXPERIMENT 6: Attention Distribution Entropy  (L=" +
                  std::to_string(L) + ", d=" + std::to_string(d) + ")");
 
     std::cout << "  Higher entropy = more diffuse attention.\n";
     std::cout << "  Sketching should preserve entropy structure.\n\n";
 
     csv.row("exp", "trial", "k", "method", "entropy");
 
     std::cout << std::left  << std::setw(24) << "Method"
               << std::right << std::setw(14) << "Mean Entropy"
               << std::setw(14) << "Std Entropy"
               << std::setw(16) << "Delta vs Exact"
               << "\n";
     std::cout << std::string(70, '-') << "\n";
 
     int trials = 10;
 
     attn::ExactAttention<Scalar>::Config ecfg;
     ecfg.compute_diag = true;
     attn::ExactAttention<Scalar> exact(ecfg);
 
     // Collect exact entropies
     std::vector<Scalar> exact_entropies;
     for (int t = 0; t < trials; ++t) {
         Matrix Q, K, V;
         gen_qkv(L, d, Q, K, V, static_cast<uint64_t>(t) * 31);
         auto res = exact.forward(Q, K, V);
         exact_entropies.push_back(res.attn_entropy);
         csv.row("entropy", t, d, "exact", res.attn_entropy);
     }
     Scalar exact_mean = std::accumulate(exact_entropies.begin(),
                                         exact_entropies.end(), Scalar{0}) / trials;
     Scalar exact_var  = Scalar{0};
     for (auto e : exact_entropies)
         exact_var += (e - exact_mean) * (e - exact_mean);
     Scalar exact_std  = std::sqrt(exact_var / (trials - 1));
 
     std::cout << std::left  << std::setw(24) << "Exact (baseline)"
               << std::right << std::fixed << std::setprecision(4)
               << std::setw(14) << exact_mean
               << std::setw(14) << exact_std
               << std::setw(16) << "0.0000"
               << "\n";
 
     struct EntM {
         const char*      tag;
         attn::SketchType type;
     };
     std::vector<EntM> ent_methods = {
         {"gauss", attn::SketchType::GAUSSIAN},
     };
     if (srht_dim_ok(d))
         ent_methods.push_back({"srht", attn::SketchType::SRHT});
 
     for (const auto& em : ent_methods) {
         for (int k : {d / 4, d / 2, d}) {
             if (k < 1) continue;
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.sketch_dim   = k;
             cfg.sketch_type  = em.type;
             cfg.compute_diag = true;
             attn::SketchedAttention<Scalar> sk(cfg);
 
             std::vector<Scalar> sk_entropies;
             for (int t = 0; t < trials; ++t) {
                 Matrix Q, K, V;
                 gen_qkv(L, d, Q, K, V, static_cast<uint64_t>(t) * 31);
                 auto res = sk.forward(Q, K, V);
                 sk_entropies.push_back(res.attn_entropy);
                 std::string mlab =
                     std::string(em.tag) + "_k=" + std::to_string(k);
                 csv.row("entropy", t, k, mlab, res.attn_entropy);
             }
 
             Scalar sk_mean = std::accumulate(sk_entropies.begin(),
                                              sk_entropies.end(), Scalar{0}) / trials;
             Scalar sk_var  = Scalar{0};
             for (auto e : sk_entropies) sk_var += (e - sk_mean) * (e - sk_mean);
             Scalar sk_std  = std::sqrt(sk_var / (trials - 1));
             Scalar delta   = sk_mean - exact_mean;
 
             std::string label =
                 std::string(em.tag) + " k=" + std::to_string(k);
             std::cout << std::left  << std::setw(24) << label
                       << std::right << std::fixed << std::setprecision(4)
                       << std::setw(14) << sk_mean
                       << std::setw(14) << sk_std
                       << std::setw(16) << (delta >= 0 ? "+" : "") << delta
                       << "\n";
         }
     }
 }
 
 // ============================================================
 // Experiment 7: Sequence-dimension sketch — speedup vs exact
 // ============================================================
 void exp7_seq_sketch(int d, CsvWriter& csv) {
     print_header("EXPERIMENT 7: Sequence-Sketch Speedup  (THE KEY RESULT)");
 
     std::vector<int> seq_lens = {512, 1024, 2048, 4096, 8192};
     std::vector<double> fractions = {1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16};
 
     std::cout << std::left  << std::setw(8)  << "L"
               << std::right << std::setw(14) << "Exact(ms)"
               << std::setw(16) << "k=L/2(ms)"
               << std::setw(10) << "Speedup"
               << std::setw(16) << "k=L/4(ms)"
               << std::setw(10) << "Speedup"
               << std::setw(16) << "k=L/8(ms)"
               << std::setw(10) << "Speedup"
               << std::setw(16) << "k=L/16(ms)"
               << std::setw(10) << "Speedup"
               << "\n";
     std::cout << std::string(100, '-') << "\n";
 
     csv.row("seq_sketch", "L", "d", "k_seq", "fraction",
             "method", "time_ms", "speedup",
             "cos_sim", "rel_err");
 
     for (int L : seq_lens) {
         Matrix Q = Matrix::Random(L, d);
         Matrix K = Matrix::Random(L, d);
         Matrix V = Matrix::Random(L, d);
 
         attn::ExactAttention<Scalar>::Config ecfg;
         attn::ExactAttention<Scalar> exact(ecfg);
         double t_exact = min_time_ms([&] { exact.forward(Q, K, V); });
         auto   exact_res = exact.forward(Q, K, V);
 
         std::cout << std::left << std::setw(8) << L
                   << std::right << std::fixed << std::setprecision(2)
                   << std::setw(14) << t_exact;
 
         csv.row("seq_sketch", L, d, L, 1.0,
                 "exact", t_exact, 1.0, 1.0, 0.0);
 
         for (double frac : fractions) {
             int k_seq = std::max(8, static_cast<int>(L * frac));
 
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.sketch_dim = d;
             cfg.rng_seed   = 42;
             attn::SketchedAttention<Scalar> sk(cfg);
 
             double t_sk = min_time_ms([&] {
                 sk.forward_seq_sketch(Q, K, V, k_seq);
             });
             auto sk_res = sk.forward_seq_sketch(Q, K, V, k_seq);
 
             double speedup = t_exact / t_sk;
             Scalar cos_sim = cosine_similarity(exact_res.output, sk_res.output);
             Scalar rel_err = rel_frob_error(exact_res.output, sk_res.output);
 
             std::cout << std::setw(16) << t_sk
                       << std::setw(10) << speedup << "x";
 
             csv.row("seq_sketch", L, d, k_seq, frac,
                     "seq_sketch", t_sk, speedup, cos_sim, rel_err);
         }
         std::cout << "\n";
     }
 
     std::cout << "\n  Quality (cosine similarity of output vs exact):\n";
     std::cout << std::left  << std::setw(8)  << "L"
               << std::right << std::setw(16) << "k=L/4"
               << std::setw(16) << "k=L/8"
               << std::setw(16) << "k=L/16"
               << "\n";
     std::cout << std::string(60, '-') << "\n";
 
     for (int L : seq_lens) {
         Matrix Q = Matrix::Random(L, d);
         Matrix K = Matrix::Random(L, d);
         Matrix V = Matrix::Random(L, d);
 
         attn::ExactAttention<Scalar>::Config ecfg;
         attn::ExactAttention<Scalar> exact(ecfg);
         auto exact_res = exact.forward(Q, K, V);
 
         std::cout << std::left << std::setw(8) << L;
 
         for (double frac : fractions) {
             int k_seq = std::max(8, static_cast<int>(L * frac));
             attn::SketchedAttention<Scalar>::Config cfg;
             cfg.rng_seed = 42;
             attn::SketchedAttention<Scalar> sk(cfg);
             auto sk_res = sk.forward_seq_sketch(Q, K, V, k_seq);
             Scalar cos_sim = cosine_similarity(exact_res.output, sk_res.output);
             std::cout << std::right << std::fixed << std::setprecision(6)
                       << std::setw(16) << cos_sim;
         }
         std::cout << "\n";
     }
 }
 
 // ============================================================
 // Main
 // ============================================================
 int main(int argc, char** argv) {
     int d_model     = (argc > 1) ? std::stoi(argv[1]) : 64;
     int max_seq_len = (argc > 2) ? std::stoi(argv[2]) : 2048;
     int num_heads   = (argc > 3) ? std::stoi(argv[3]) : 1;
 
     // For single-head experiments, d = d_model
     int d = d_model / std::max(1, num_heads);
 
     std::cout << "\n";
     std::cout << std::string(90, '=') << "\n";
     std::cout << "  Sketched Attention Benchmark\n";
     std::cout << std::string(90, '=') << "\n";
     std::cout << "  d_model     : " << d_model     << "\n";
     std::cout << "  d per head  : " << d           << "\n";
     std::cout << "  max_seq_len : " << max_seq_len << "\n";
     std::cout << "  num_heads   : " << num_heads   << "\n";
     std::cout << std::string(90, '=') << "\n";
 
     CsvWriter csv("attention_benchmark_results.csv");
 
     // Experiment 1: scaling vs L
     exp1_scaling(d, num_heads, csv);
 
     // Experiment 2: quality vs k  (fixed L = max_seq_len/2)
     exp2_sketch_size(std::min(max_seq_len / 2, 1024), d, csv);
 
     // Experiment 3: sketch type comparison
     exp3_sketch_types(std::min(max_seq_len / 2, 512), d, csv);
 
     // Experiment 4: gradient fidelity
     exp4_gradient_fidelity(256, d, csv);
 
     // Experiment 5: multi-head
     if (d_model >= 64)
         exp5_multihead(std::min(max_seq_len / 4, 512), d_model, csv);
 
     // Experiment 6: entropy
     exp6_entropy(256, d, csv);
 
    //  Experiment 7: sequence sketch speedup
     exp7_seq_sketch(d, csv);
 
     std::cout << "\n[CSV] All results written to attention_benchmark_results.csv\n";
 
     std::cout << "\nQuick plot command:\n";
     std::cout << R"(python3 - << 'EOF'
 import pandas as pd
 import matplotlib.pyplot as plt
 
 df = pd.read_csv('attention_benchmark_results.csv')
 
 fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
 # Plot 1: scaling
 s = df[df['exp']=='scaling']
 for m, g in s.groupby('method'):
     axes[0].plot(g['L'], g['time_ms'], marker='o', label=m)
 axes[0].set_xlabel('Sequence Length L')
 axes[0].set_ylabel('Time (ms)')
 axes[0].set_title('Attention Time vs Sequence Length')
 axes[0].legend(); axes[0].set_yscale('log')
 
 # Plot 2: quality ablation
 a = df[df['exp']=='ablation']
 axes[1].plot(a['k_over_d'], a['output_cos_sim'], marker='s', color='C1')
 axes[1].axhline(1.0, linestyle='--', color='C0', label='Exact')
 axes[1].set_xlabel('k / d'); axes[1].set_ylabel('Cosine Similarity')
 axes[1].set_title('Output Quality vs Sketch Size')
 axes[1].legend()
 
 # Plot 3: fidelity
 f = df[df['exp']=='fidelity']
 for k, g in f.groupby('k'):
     axes[2].hist(g['cos_sim'], alpha=0.5, label=f'k={k}', bins=10)
 axes[2].set_xlabel('Cosine Similarity')
 axes[2].set_title('Output Fidelity Distribution')
 axes[2].legend()
 
 plt.tight_layout()
 plt.savefig('attention_benchmark.png', dpi=150)
 print('Saved attention_benchmark.png')
 EOF
 )";
     std::cout << "\n";
 
     return 0;
 }