from pathlib import Path

from setuptools import find_packages, setup

readme = Path("README.md").read_text(encoding="utf-8")
requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

setup(
    name="vico-lite",
    version="0.3.0",
    description=(
        "Training-free embodied multi-agent cooperation on ThreeDWorld "
        "Multi-Agent Transport, bundled with the TDW transport challenge controller."
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/moo-duck-bit/vico_lite_RA-L",
    keywords="embodied-ai multi-agent tdw simulation clip",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"vico_lite.language": ["*.txt"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
)
