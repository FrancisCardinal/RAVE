import setuptools

setuptools.setup(
    name="pycount",
    version="dev",
    author="introlab",
    author_email="",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/introlab/demo_integration",
    project_urls={
        "Bug Tracker": "https://github.com/introlab/demo_integration/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
