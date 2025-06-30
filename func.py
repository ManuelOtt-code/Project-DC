from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

# additional imports from "From_AODB" for function migration
import re
import os
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
import math
from sklearn.manifold import TSNE


def parse_experimental_data(row):
    # Use regex to extract components
    match = re.match(r"(IC50|EC50)\s*(>=|>|=)\s*([\d.]+)\s*(nM|ug\.mL-1|µg\.mL-1)", row)
    if match:
        return {
            "activity_type": match.group(1),
            "operator": match.group(2),
            "activity_value": float(match.group(3)),
            "units": match.group(4).replace(
                ".", ""
            ),  # Remove dots (e.g., ug.mL-1 -> ugmL-1)
        }
    else:
        return None


# convert units of ugmL-1 activity values in nM
def convert_units(row):
    """Proper unit conversion with consistent numeric output"""
    # Check if conversion is needed and possible
    if (
        row["units"] == "ugmL-1"
        and "Molecular Weight" in row
        and not pd.isna(row["Molecular Weight"])
        and not pd.isna(row["activity_value"])
    ):

        # Convert µg/mL to nM: (µg/mL * 10^6) / (g/mol) = nM
        return round(
            (float(row["activity_value"]) * 10**6) / float(row["Molecular Weight"])
        )

    # Return original value if already in nM or no conversion possible
    elif row["units"] == "nM" and not pd.isna(row["activity_value"]):
        return float(row["activity_value"])

    # Return NaN for unconvertable cases
    return np.nan


# Convert IC50 values in nM into pIC50 values
def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)  # 9 for nano molar
    return pIC50_value


def canonicalize(smiles):
    try:

        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print("Smiles", smiles)
        return np.nan

    if mol is None:
        return np.nan  # Handle invalid SMILES by returning None
    canonical_smiles = Chem.MolToSmiles(mol)
    return canonical_smiles


def is_substruct_in(mol_smiles, substruct_smarts):
    mol = Chem.MolFromSmiles(mol_smiles)
    mol_substruct = Chem.MolFromSmarts(substruct_smarts)
    if mol is None:
        return None  # Handle invalid SMILES by returning None

    match = mol.HasSubstructMatch(mol_substruct)
    if match:
        return 1
    else:
        return 0


def get_TanimotoSimilarity(smiles_mol1, smiles_mol2):
    if smiles_mol1 is None or smiles_mol2 is None:
        return np.nan
    fp1 = get_MorganFingerprint(smiles_mol=smiles_mol1)
    fp2 = get_MorganFingerprint(smiles_mol=smiles_mol2)
    if fp1 is np.nan or fp2 is np.nan:
        return np.nan
    Similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return Similarity


def get_MorganFingerprint(smiles_mol):
    if smiles_mol is None or smiles_mol is np.nan or smiles_mol == "":
        return np.nan
    rd_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    mol = Chem.MolFromSmiles(smiles_mol)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)

    return fp


def convert_smiles_series_to_fp_to_np_array(smiles_series):
    fps = smiles_series.apply(get_MorganFingerprint)
    fps = fps.tolist()
    fp_array = []
    for fp in fps:
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array.append(arr)
    return np.array(fp_array)


def transform_index(smiles1, smiles2, index):
    if smiles1 == smiles2:
        return index


def count_subgroup_number(subgroup_smarts, smiles_mol):
    mol_of_interest = Chem.MolFromSmiles(smiles_mol)
    OH = Chem.MolFromSmarts(subgroup_smarts)
    subgroup_pattern = mol_of_interest.GetSubstructMatches(OH)
    return len(subgroup_pattern)


def linear_slope(m, x, a):
    return m * x + a


def heatmap_corr(corr_matrix, plottitle, plotname):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(plottitle)
    plt.savefig(plotname)
    plt.clf()


def combine_lit_assays(column_1, column_2):
    df_combined = pd.DataFrame()
    df_combined["Lit_1"] = column_1
    df_combined["Lit_2"] = column_2
    print(df_combined)
    df_combined = df_combined.dropna()
    return df_combined
