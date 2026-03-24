from setuptools import setup, find_packages

setup(
    name="language_conditioned_unigo2",
    version="0.1.0",
    description="Language-conditioned whole-body contact manipulation for Unitree Go2",
    authors=["Hongkun Gong", "Hanzhi Bian"],
    packages=find_packages(),
    python_requires=">=3.11",
)