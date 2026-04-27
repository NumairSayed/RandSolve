#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "lowrank/RandomizedSVD.hpp"
#include "rng/StdRNG.hpp"   // ✅ correct include

namespace py = pybind11;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

py::tuple rsvd(const Matrix& A, int r, const std::string& sketch) {

    int oversampling = 10;
    int n_iter = 2;

    randnla::RandomizedSVD<double> rsvd(oversampling, n_iter);

    randnla::StdRNG<double> rng(42);   // ✅ seed for reproducibility

    rsvd.compute(A, r, rng, sketch);

    Matrix U = rsvd.matrixU().leftCols(r);
    Vector S = rsvd.singularValues().head(r);

    Matrix Vt = S.asDiagonal().inverse() * U.transpose() * A;

    return py::make_tuple(U, S, Vt);
}

PYBIND11_MODULE(rsvd_backend, m) {
    m.def("rsvd", &rsvd);
}