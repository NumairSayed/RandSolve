/**
 * @file torch_extension.cpp
 * @brief PyTorch C++ extension exposing RandNLA primitives.
 *
 * Build with:
 *   python setup.py build_ext --inplace
 *
 * Then in Python:
 *   import randnla_ext
 *   U, S, Vt = randnla_ext.randomized_svd(W_tensor, rank=8, oversample=10)
 */
 #include <numeric>      // std::iota
 #include <random>       // std::mt19937
 #include <algorithm>    // std::swap
 #include <cstring>      // std::memcpy
 
 #include <torch/extension.h>
 #include <Eigen/Dense>
 
 
 #include <torch/extension.h>
 #include <Eigen/Dense>
 #include "lowrank/RandomizedSVD.hpp"
 #include "rng/StdRNG.hpp"
 #include "core/Precision.hpp"
 
 // ── Tensor ↔ Eigen conversion ─────────────────────────────────────────────
 
 using EMatrix = randnla::PrecisionTraits<float>::Matrix;
 
 EMatrix tensor_to_eigen(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 2, "Expected 2D tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(!t.is_cuda(), "Expected CPU tensor");

    // CRITICAL: PyTorch is row-major (C order), Eigen default is col-major.
    // We must use Map with correct strides instead of memcpy.
    auto t_cont = t.contiguous();
    int m = t_cont.size(0), n = t_cont.size(1);

    // Map PyTorch's row-major buffer into Eigen with explicit stride
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>>
        mapped(t_cont.data_ptr<float>(), m, n);

    // Convert to Eigen's default col-major for correct BLAS operations
    EMatrix M = mapped;
    return M;
}

torch::Tensor eigen_to_tensor(const EMatrix& M) {
    // Write back as row-major so PyTorch reads it correctly
    int m = M.rows(), n = M.cols();
    auto t = torch::zeros({m, n}, torch::kFloat32);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>>
        mapped(t.data_ptr<float>(), m, n);
    mapped = M;  // converts col-major → row-major on copy
    return t;
}
 
 // ── Exposed functions ─────────────────────────────────────────────────────
 
 /**
  * Randomized SVD.
  * Returns (U, S, Vt) where W ≈ U * diag(S) * Vt
  * U shape: (m, rank), S shape: (rank,), Vt shape: (rank, n)
  */
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 randomized_svd(const torch::Tensor& W,
                int rank,
                int oversample = 10,
                int power_iters = 2,
                int seed = 42) {
     EMatrix A = tensor_to_eigen(W);
     randnla::StdRNG<float> rng(static_cast<uint64_t>(seed));
     randnla::RandomizedSVD<float> rsvd(oversample, power_iters);
     rsvd.compute(A, rank, rng);
 
     torch::Tensor U  = eigen_to_tensor(rsvd.matrixU());
     torch::Tensor Vt = eigen_to_tensor(rsvd.matrixVt());
 
     // Singular values → 1D tensor
     auto sv = rsvd.singularValues();
     torch::Tensor S = torch::zeros({rank}, torch::kFloat32);
     for (int i = 0; i < rank; ++i)
         S[i] = sv(i);
 
     return {U, S, Vt};
 }
 
 /**
  * Sketched matrix multiply: (SQ)(SK)^T
  * Used for benchmarking sketched attention scores.
  * Returns approximate QK^T of shape (L, L).
  */
 torch::Tensor sketched_attention_scores(const torch::Tensor& Q,
                                          const torch::Tensor& K,
                                          int k_seq,
                                          int seed = 42) {
     TORCH_CHECK(Q.dim() == 2 && K.dim() == 2, "Expected 2D tensors");
     int L = Q.size(0), d = Q.size(1);
     k_seq = std::min(k_seq, L);
 
     EMatrix Qe = tensor_to_eigen(Q);
     EMatrix Ke = tensor_to_eigen(K);
 
     // Row-sample k_seq landmark indices
     std::vector<int> idx(L);
     std::iota(idx.begin(), idx.end(), 0);
     std::mt19937 gen(static_cast<uint32_t>(seed));
     for (int i = 0; i < k_seq; ++i) {
         std::uniform_int_distribution<int> dist(i, L-1);
         std::swap(idx[i], idx[dist(gen)]);
     }
 
     // Gather landmark rows
     EMatrix K_land(k_seq, d), dummy(k_seq, d);
     for (int i = 0; i < k_seq; ++i)
         K_land.row(i) = Ke.row(idx[i]);
 
     // Scores: (L, k_seq)
     float scale = 1.0f / std::sqrt(float(d));
     EMatrix Scores = (Qe * K_land.transpose()) * scale;
 
     return eigen_to_tensor(Scores);  // (L, k_seq) — caller applies softmax
 }
 
 // ── pybind11 module ───────────────────────────────────────────────────────
 
 PYBIND11_MODULE(randnla_ext, m) {
     m.doc() = "RandNLA C++ backend for PyTorch";
 
     m.def("randomized_svd", &randomized_svd,
           "Randomized SVD: W ≈ U diag(S) Vt",
           py::arg("W"),
           py::arg("rank"),
           py::arg("oversample")   = 10,
           py::arg("power_iters")  = 2,
           py::arg("seed")         = 42);
 
     m.def("sketched_attention_scores", &sketched_attention_scores,
           "Landmark attention scores (L, k_seq)",
           py::arg("Q"),
           py::arg("K"),
           py::arg("k_seq"),
           py::arg("seed") = 42);
 }