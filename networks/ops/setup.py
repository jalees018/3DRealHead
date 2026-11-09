"""Build the StyleGAN2 CUDA ops: `cd networks/ops && pip install .` (or `python setup.py install`)."""
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="stylegan2_ops",
    ext_modules=[
        cpp_extension.CUDAExtension("fused", ["fused_bias_act.cpp", "fused_bias_act_kernel.cu"]),
        cpp_extension.CUDAExtension("upfirdn2d", ["upfirdn2d.cpp", "upfirdn2d_kernel.cu"]),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
