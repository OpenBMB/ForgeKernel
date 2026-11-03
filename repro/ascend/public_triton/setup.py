from setuptools import find_packages, setup


setup(
    name="cann_bench",
    version="1.0.0",
    description="Public Triton-Ascend Exp and SwiGlu reproduction for CANN Bench",
    packages=find_packages(),
    python_requires=">=3.10",
)
