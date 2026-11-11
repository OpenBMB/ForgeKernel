from setuptools import setup

setup(
    name="cann_bench",
    version="1.0.0",
    packages=["cann_bench", "cann_bench.swi_glu"],
    package_data={
        "cann_bench.swi_glu": [
            "c1/*.py",
            "c2/*.py",
            "c3/*.py",
            "c4/*.py",
            "c5/*.py",
            "c6/*.py",
            "c7/*.py",
            "c8/*.py",
        ],
    },
    python_requires=">=3.8",
)
