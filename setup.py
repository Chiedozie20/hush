from setuptools import setup, find_packages

setup(
    name="hush",
    version="0.1.0",
    description="FPGA-accelerated Conv1d for Whisper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)