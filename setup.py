from setuptools import setup, find_packages

setup(
    name='transformerdock',
    version='0.1.0',
    description='Transformer-based protein-protein docking scoring function',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'torch-geometric>=2.0.0',
        'torch-scatter',
        'numpy',
        'scipy',
        'scikit-learn',
        'plyfile',
        'biopython',
        'open3d',
        'requests',
        'pandas',
    ],
)
