cmake_minimum_required(VERSION 3.16)
project(randnla_cpp VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# C++17
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# Options
# ============================================================
option(RANDNLA_USE_LAPACK      "Link against system BLAS/LAPACK" OFF)
option(RANDNLA_BUILD_EXPERIMENTS "Build speed experiment harness" ON)

# ============================================================
# Compiler flags
# ============================================================
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0)
endif()
add_link_options(
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Debug>>:-fsanitize=address>
)

# ============================================================
# Eigen3
# ============================================================
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# ============================================================
# Optional BLAS/LAPACK
#   Enable with:  cmake .. -DRANDNLA_USE_LAPACK=ON
#   Eigen will automatically use BLAS/LAPACK internally when
#   EIGEN_USE_BLAS / EIGEN_USE_LAPACKE are defined.
# ============================================================
if(RANDNLA_USE_LAPACK)
  find_package(BLAS   REQUIRED)
  find_package(LAPACK REQUIRED)
  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -DRANDNLA_USE_LAPACK)
  message(STATUS "[RandNLA] BLAS/LAPACK backend: ON")
else()
  message(STATUS "[RandNLA] Eigen-only backend: ON  (use -DRANDNLA_USE_LAPACK=ON to enable BLAS)")
endif()

# ============================================================
# Shared include path
# ============================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================
# Helper function to link a target
# ============================================================
function(randnla_link_target target)
  target_link_libraries(${target}
    Eigen3::Eigen
    $<$<BOOL:${RANDNLA_USE_LAPACK}>:BLAS::BLAS>
    $<$<BOOL:${RANDNLA_USE_LAPACK}>:LAPACK::LAPACK>
  )
endfunction()

# ============================================================
# Main demo
# ============================================================
add_executable(randnla_main src/main.cpp src/utils.cpp)
randnla_link_target(randnla_main)

# ============================================================
# Speed experiment
# ============================================================
if(RANDNLA_BUILD_EXPERIMENTS)
  add_executable(speed_experiment src/speed_experiment.cpp src/utils.cpp)
  randnla_link_target(speed_experiment)
endif()

# ============================================================
# Install
# ============================================================
install(TARGETS randnla_main DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/randnla)

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "==============================")
message(STATUS "  RandNLA Build Configuration")
message(STATUS "  Type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Standard    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  LAPACK      : ${RANDNLA_USE_LAPACK}")
message(STATUS "  Experiments : ${RANDNLA_BUILD_EXPERIMENTS}")
message(STATUS "==============================")
message(STATUS "  Quick start:")
message(STATUS "    mkdir build && cd build")
message(STATUS "    cmake .. && make -j\$(nproc)")
message(STATUS "    ./randnla_main")
message(STATUS "    ./speed_experiment")
message(STATUS "==============================")