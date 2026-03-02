# Clone / extract project
cd randnla_cpp

# Install Eigen3 (if not installed)
sudo apt install libeigen3-dev   # Ubuntu/Debian

# Build (Eigen-only, fastest setup)
make                          # uses the Makefile wrapper

# Or manually with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run demos
./randnla_main
./speed_experiment

# Run speed experiment with custom sizes
./speed_experiment 20000 50 8    # m=20000, n=50, sketch=8x

# Build with BLAS/LAPACK backend
make lapack
# or: cmake .. -DRANDNLA_USE_LAPACK=ON && make -j$(nproc)

# Debug build (with AddressSanitizer)
make debug

