# setup.py
from torch.utils.cpp_extension import CppExtension, BuildExtension
import subprocess, os
from setuptools import setup

# Absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Get Eigen path
try:
    eigen_path = subprocess.check_output(
        ["pkg-config", "--cflags-only-I", "eigen3"]
    ).decode().strip().replace("-I", "").strip()
except Exception:
    eigen_path = "/usr/include/eigen3"  # fallback — already in your compile log

setup(
    name="randnla_ext",
    ext_modules=[
        CppExtension(
            name="randnla_ext",
            sources=[os.path.join(PROJECT_ROOT, "bindings/torch_extension.cpp")],
            include_dirs=[
                os.path.join(PROJECT_ROOT, "include"),   # ← absolute path
                eigen_path,
            ],
            extra_compile_args=[
                "-O3", "-std=c++17", "-march=native",
                "-Wno-unused-parameter",   # suppress Eigen/torch warnings
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)