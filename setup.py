import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='shadowspy',
    version='0.0.1',
    author='Stefano Bertone',
    author_email='sbertone@umd.edu',
    description='A collection of tools for the illumination of planetary surfaces leveraging ray tracing techniques',
    url='',
    # packages=['shadowspy'],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.9',
    install_requires=[],
)
