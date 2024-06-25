import setuptools 
from spatialmeta._version import version

version = version
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="spatialMETA",
    version=version,
    url="https://github.com/RuonanTian/SpatialMETA",
    author="Ruonan Tian",
    author_email="ruonan.23@intl.zju.edu.cn",
    description="spatialMETA: a deep learning framework for spatial multiomics",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(
        exclude=["*docs*"]
    ),
    install_requires=[
        "sphinx_rtd_theme==1.2.0",
        "sphinx-hoverxref==1.3.0",
        "anndata==0.8.0",
        "scanpy==1.8.1",
        "pandas==1.4.2",
        "matplotlib==3.5.2",
        "seaborn==0.11.2",
        "scikit-learn==0.24.1",
        "einops==0.4.1",
        "biopython==1.79",
        "plotly==5.10.0",
        "scipy==1.10.0",
        "pyimzml==1.5.3",
        "dash_bootstrap_components==1.5.0",
        "dash_daq==0.5.0",
        "python_dotplot",
        "intervaltree==3.1.0",
        "leidenalg",
        "molmass",
        "svgpathtools==1.6.1",
        "shapely==2.0.3",
        "numpy==1.21.6",
        "numba==0.57.1",
        "umap-learn==0.5.1",
        "adjusttext==1.1.1",
        "colour==0.1.5",
        "kornia==0.7.2",
        "elasticdeform",
        "tifffile",
    ],
    extras_require={
        "gpu": [
            "torch==2.2.2+cu121",
            "torchvision==0.17.2+cu121",
            "torchgeometry==0.1.2+cu121",
        ],
        "cpu": [
            "torch==2.2.2",
            "torchvision==0.17.2",
            "torchgeometry==0.1.2",
        ]
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu121",
        "https://miropsota.github.io/torch_packages_builder",
    ],
    include_package_data=False,
)
