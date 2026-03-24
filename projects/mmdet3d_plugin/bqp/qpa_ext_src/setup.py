from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qpa_ext",
    ext_modules=[
        CUDAExtension(
            name="qpa_ext",
            sources=[
                "src/bindings.cpp",
                "src/qpa.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

