from setuptools import setup, find_packages

setup(
    name="adapt_ts_project",
    version="0.1.0",
    description="Combination of TS-TCC and AdaptNAS for time-series anomaly detection",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.12.0",
        "torchvision",
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.8",
)
