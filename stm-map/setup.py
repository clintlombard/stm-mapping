from setuptools import find_packages, setup

setup(
    name="stmmap",
    description="Stochastic Triangular Mesh Map",
    classifiers=["Programming Language :: Python"],
    keywords="python robotics mapping",
    author="Clint Lombard",
    python_requires=">= 3.7",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[x.strip() for x in open("requirements.txt").readlines()],
    version="0.0.1",
)
