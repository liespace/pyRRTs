import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rrts',
    version='1.0.8',
    packages=setuptools.find_packages(),
    description='RRTs Planners',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gabel Liemann',
    author_email='troubleli233@gmail.com',
    url='https://github.com/liespace/pyRRTs',
    license='MIT',
    install_requires=[
        'numpy',
        'numba',
        'reeds_shepp',
        'matplotlib',
        'opencv-python',
        'scipy'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
)
