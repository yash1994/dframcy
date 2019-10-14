from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    REQUIRES = [line.strip("\n") for line in f]

setup(
    name="dframcy",
    version="0.1.0",
    description="Pandas Dataframe integration for spaCy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yash1994/dframcy",
    author="Yash Patadia",
    author_email="yash@patadia.org",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords=["spacy", "dataframe", "pandas"],
    packages=find_packages(),
    install_requires=REQUIRES,
    tests_require=['pytest'],
    entry_points={
        "console_scripts": ["dframcy=dframcy.cli:main"],
    }
)
