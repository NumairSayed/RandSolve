# RandNLA C++ — Randomized Numerical Linear Algebra Scaffold

A research-grade, modular, object-oriented C++17 framework for experimenting
with Randomized Numerical Linear Algebra (RandNLA) methods.

## Architecture
```
┌──────────────────────────────────────────────────────┐
│                    Drivers Layer                     │
│  LeastSquaresDriver  │  LowRankDriver               │
└───────────────┬──────┴──────────────────────────────┘
                │ uses
┌───────────────▼──────────────────────────────────────┐
│              Randomized Algorithms                   │
│  SketchedLeastSquares  │  PreconditionedLS           │
│  ConfidenceBounds                                    │
└──────┬────────────────────────┬───────────────────────┘
       │ uses                   │ uses
┌──────▼──────┐          ┌──────▼──────┐
│  Sketching  │          │   Solvers   │
│  ISketch    │          │ ILeastSq.   │
│  Gaussian   │          │ QRSolver    │
│  Sparse     │          │ NormalEq.   │
│  SRHT       │          │ Iterative   │
└──────┬──────┘          └─────────────┘
       │ uses
┌──────▼──────┐
│    RNG      │
│   IRNG      │
│  StdRNG     │
│  EigenRNG   │
└─────────────┘
```

### Layer Separation Philosophy

| Layer | Responsibility | Depends On |
|---|---|---|
| Drivers | Route problem to correct algorithm | Randomized + Solvers |
| Randomized | Compose sketch + solver | Sketching, Solvers |
| Sketching | Apply random embedding | RNG only |
| Solvers | Deterministic linear algebra | Eigen only |
| RNG | Random variate generation | stdlib / Eigen |
| Core | Traits, decision logic | nothing |

## Build

### Prerequisites

- C++17 compiler (GCC ≥ 10, Clang ≥ 12)
- CMake ≥ 3.16
- Eigen3 ≥ 3.4  (`sudo apt install libeigen3-dev`)

### Quick start
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./randnla_main
./speed_experiment
```

### With BLAS/LAPACK backend
```bash
cmake .. -DRANDNLA_USE_LAPACK=ON
make -j$(nproc)
```

When `-DRANDNLA_USE_LAPACK=ON`, the preprocessor defines
`EIGEN_USE_BLAS` and `EIGEN_USE_LAPACKE`, so Eigen routes dense
decompositions through your system BLAS/LAPACK automatically —
**no code changes required**.

## Switching Precision

Every algorithm is templated on `Scalar`:
```cpp
// double precision (default)
SketchedLeastSquares<double> sls(sketch, solver);

// single precision — useful for GPU pipelining
SketchedLeastSquares<float> sls_f(sketch_f, solver_f);
```

## Adding a New Sketch

1. Create `include/sketching/MySketch.hpp`
2. Inherit from `ISketch<Scalar>`
3. Implement `apply(Matrix)` and `apply(Vector)`
4. Only accept `IRNG<Scalar>&` for randomness
```cpp
template<typename Scalar>
class MySketch : public ISketch<Scalar> {
public:
    MySketch(int k, IRNG<Scalar>& rng) : k_(k), rng_(rng) {}
    Matrix apply(const Matrix& A) override { /* ... */ }
    Vector apply(const Vector& b) override { /* ... */ }
private:
    int k_;
    IRNG<Scalar>& rng_;
};
```

## Adding a New Solver

1. Create `include/solvers/MySolver.hpp`
2. Inherit from `ILeastSquaresSolver<Scalar>`
3. Implement `solve(A, b)` — **no sketching dependency allowed**

## References

- Mahoney (2011) *Randomized Algorithms for Matrices and Data*
- Woodruff (2014) *Sketching as a Tool for NLA*
- Murray, Demmel et al. (2023) *Randomized Numerical Linear Algebra*
- RandBLAS (https://github.com/BallisticLA/RandBLAS)