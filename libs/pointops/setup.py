import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
if opt:
    os.environ["OPT"] = " ".join(
        flag for flag in opt.split() if flag != "-Wstrict-prototypes"
    )

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

nvcc_args = ["-O2"]
msvc_bin = os.environ.get("MSVC_HOSTX64_BIN")
if msvc_bin and os.path.exists(os.path.join(msvc_bin, "cl.exe")):
    nvcc_args += ["-ccbin", msvc_bin]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops", "pointops.functions"],
    package_dir={"pointops": "."},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={"cxx": [], "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
