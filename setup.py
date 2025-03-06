from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="onelinerml",
    version="0.1.2",
    description="A one-liner ML training and deployment library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexis Jean Baptiste",
    author_email="your.email@example.com",
    url="https://github.com/alexiisjbaptiste/onelinerml",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "fastapi",
        "uvicorn",
        "joblib",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "onelinerml-train=onelinerml.train:main",
            "onelinerml-deploy=onelinerml.deploy:main",
        ],
    },
)
