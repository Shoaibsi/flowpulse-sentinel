from setuptools import setup, find_packages

setup(
    name="flowpulse",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "duckdb",
        "python-telegram-bot",
        "websockets",
        "requests",
        "pandas-ta",
        "yfinance",
        "polygon-api-client",
        "google-generativeai"
    ]
)
