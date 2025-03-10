from setuptools import setup, find_packages

setup(
    name="onelinerml",
    version="0.1.3",
    description="A one-line machine learning library with remote API and dashboard support.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "streamlit",
        "pyngrok"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "onelinerml-serve=onelinerml.serve:main"
        ]
    },
)
