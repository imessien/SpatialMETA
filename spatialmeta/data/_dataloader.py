import os
from pathlib import Path
import scanpy as sc
import pandas as pd
import warnings
import subprocess
from typing import Literal, Union
from pyimzml.ImzMLParser import ImzMLParser
from ..util._classes import AnnDataST, AnnDataSM, AnnDataJointSMST
MODULE_PATH = Path(__file__).parent
warnings.filterwarnings("ignore")

zenodo_accession = '14986870'
zenodo_file_path = pd.read_csv(os.path.join(MODULE_PATH, 'zenodo_url.txt'), sep='\t')

def list_datasets():
    """
    List all available datasets in the package.
    """
    return zenodo_file_path


def load_imzML_and_ibd(sample_name: str) -> ImzMLParser:
    """
    Load the imzML file for the given sample name.

    :param sample_name: str
        The name of the sample. Use `list_datasets` to get the list of all available datasets.
    """
    valid_sample_names = set(
        list(
            map(
                lambda x: x.split(".ibd")[0],
                filter(lambda x: x.endswith("ibd"), zenodo_file_path["file_name"]),
            )
        )
    )
    if sample_name not in valid_sample_names:
        raise ValueError(
            f"Invalid sample name. Valid sample names are {valid_sample_names}"
        )
    default_path_imzml = os.path.join(MODULE_PATH, f"./datasets/{sample_name}.imzML")
    default_path_ibd = os.path.join(MODULE_PATH, f"./datasets/{sample_name}.ibd")

    if os.path.exists(default_path_imzml) and os.path.exists(default_path_ibd):
        return ImzMLParser(default_path_imzml)
    else:
        import subprocess
        print(f"Downloading from https://zenodo.org/records/{zenodo_accession}/files/{sample_name}.imzML?download=1")
        ret1 = subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                default_path_imzml,
                f"https://zenodo.org/records/{zenodo_accession}/files/{sample_name}.imzML?download=1",
            ],
            check=True,
        )
        print(f"Downloading from https://zenodo.org/records/{zenodo_accession}/files/{sample_name}.ibd?download=1")
        ret2 = subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                default_path_ibd,
                f"https://zenodo.org/records/{zenodo_accession}/files/{sample_name}.ibd?download=1",
            ],
            check=True,
        )
        if ret1.returncode == 0 and ret2.returncode == 0:
            try:
                return ImzMLParser(default_path_imzml)
            except Exception as e:
                raise RuntimeError("Failed to download the dataset.")
        else:
            raise RuntimeError("Failed to download the dataset.")


def load_adata(
    sample_name: str,
    modality: Literal["ST", "SM", "joint"],
) -> Union[AnnDataST, AnnDataSM, AnnDataJointSMST]:
    """
    Load the AnnData object for the given sample name and modality.

    :param sample_name: str
        The name of the sample. Use `list_datasets` to get the list of all available datasets.
    :param modality: Literal["ST", "SM", "joint"]
        The modality of the dataset. Choose from "ST", "SM", or "joint".
    """
    valid_sample_names = set(
        list(
            map(
                lambda x: "_".join(x.split(".h5ad")[0].split("_")[2:]),
                filter(lambda x: x.endswith("h5ad"), zenodo_file_path["file_name"]),
            )
        )
    )
    if sample_name not in valid_sample_names:
        raise ValueError(
            f"Invalid sample name. Valid sample names are {valid_sample_names}"
        )

    default_path_h5ad = os.path.join(
        MODULE_PATH, f"./datasets/adata_{modality}_{sample_name}_raw.h5ad"
    )
    
    print(default_path_h5ad)
    if os.path.exists(default_path_h5ad):
        return sc.read(default_path_h5ad)
    else:
        import subprocess
        print(f"Downloading from https://zenodo.org/records/{zenodo_accession}/files/adata_{modality}_{sample_name}.h5ad?download=1")
        ret = subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                default_path_h5ad,
                f"https://zenodo.org/records/{zenodo_accession}/files/adata_{modality}_{sample_name}.h5ad?download=1",
            ],
            check=True,
        )
        if ret.returncode == 0:
            try:
                if modality == "ST":
                    return AnnDataST.from_anndata(sc.read(default_path_h5ad))
                elif modality == "SM":
                    return AnnDataSM.from_anndata(sc.read(default_path_h5ad))
                elif modality == "joint":
                    return AnnDataJointSMST.from_anndata(sc.read(default_path_h5ad))
            except Exception as e:
                raise RuntimeError("Failed to download the dataset.")
        else:
            raise RuntimeError("Failed to download the dataset.")


def load_Vicari_2024_msi() -> pd.DataFrame:
    """
    Load the mouse 3 MSI data from Vicari et al., 2024.
    """

    default_path_msi = os.path.join(MODULE_PATH, f'./datasets/mouse3_brain_msi.csv')
    if os.path.exists(default_path_msi):
        return pd.read_csv(default_path_msi)
    else:
        import subprocess
        print(f"Downloading from https://zenodo.org/records/{zenodo_accession}/files/mouse3_brain_msi.csv?download=1")
        ret = subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                default_path_msi,
                f"https://zenodo.org/records/{zenodo_accession}/files/mouse3_brain_msi.csv?download=1",
            ],
            check=True,
        )
        if ret.returncode == 0:
            try:
                return pd.read_csv(default_path_msi)
            except Exception as e:
                raise RuntimeError("Failed to download the dataset.")
        else:
            raise RuntimeError("Failed to download the dataset.")
