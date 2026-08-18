from setuptools import find_packages, setup


setup(
    name="nas-ade-uad",
    version="1.0.0",
    description="NAS-ADE experiments for unsupervised time-series anomaly detection",
    packages=find_packages(include=("src", "src.*")),
    install_requires=[
        "torch==2.7.1",
        "numpy==2.4.3",
        "scikit-learn==1.8.0",
        "matplotlib==3.10.8",
        "pandas==3.0.1",
        "scipy==1.17.1",
        "einops==0.8.1",
    ],
    python_requires=">=3.10,<3.13",
)
