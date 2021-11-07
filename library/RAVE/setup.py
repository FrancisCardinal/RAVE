import setuptools

setuptools.setup(
    name="rave-JacobKealey",
    version="dev",
    author="Jacob Kealey",
    author_email="jacob.kealey@usherbrooke.ca",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/FrancisCardinal/RAVE",
    project_urls={
        "Bug Tracker": "https://github.com/FrancisCardinal/RAVE/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "setuptools>=42",
        "wheel",
        "matplotlib",
        "numpy",
        "opencv-python",
        "tqdm",
        "torch",
        "torchaudio",
        "torchvision",
    ],
)
