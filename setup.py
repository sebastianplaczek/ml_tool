from setuptools import setup, find_packages

INSTALL_REQUIRES = ["pandas==2.2.2"]

setup(
    name="ml_tool",
    version="0.0.1",
    description="Tool for machine learning",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    author="Seb@",
    author_email="sebastian.placzek.af@gmail.com",
)
