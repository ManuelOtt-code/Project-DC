
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

#additional imports from "From_AODB" for function migration
import re
import os
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
import math
from sklearn.manifold import TSNE




def parse_experimental_data(row):
    # Use regex to extract components
    match = re.match(r'(IC50|EC50)\s*(>=|>|=)\s*([\d.]+)\s*(nM|ug\.mL-1|µg\.mL-1)', row)
    if match:
        return {
            'activity_type': match.group(1),
            'operator': match.group(2),
            'activity_value': float(match.group(3)),
            'units': match.group(4).replace('.', '')  # Remove dots (e.g., ug.mL-1 -> ugmL-1)
        }
    else:
        return None



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
    mol= Chem.MolFromSmiles(smiles_mol)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
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



def convert_smiles_series_to_fp_to_np_array_exception_handling(smiles_series):
    """
    Converts a pandas Series of SMILES strings into a NumPy array of Morgan fingerprints.
    
    Args:
        smiles_series (pd.Series): A pandas Series containing SMILES strings.

    Returns:
        np.ndarray: A 2D NumPy array where each row is a Morgan fingerprint.
    """
    # Initialize an empty list to store fingerprint arrays
    fp_array = []

    for smiles in smiles_series:
        if not smiles or smiles is np.nan:
            # Handle empty or invalid SMILES
            fp_array.append(np.zeros(1024, dtype=np.int8))
            continue
        
        try:
            # Generate Morgan fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Handle invalid SMILES
                fp_array.append(np.zeros(1024, dtype=np.int8))
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_array.append(arr)
        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"Error processing SMILES: {smiles}. Exception: {e}")
            fp_array.append(np.zeros(1024, dtype=np.int8))
    
    # Convert the list of arrays to a NumPy array
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
    return m*x + a



def plot_scatter(df_column_1, df_column_2, plottitle=None, plotname=None, x_axis_title=None, y_axis_title=None, ylim=None, xlim=None):
    plt.figure(figsize=(10, 10))
    if plottitle !=None:
        plt.title(plottitle)
    if x_axis_title !=None:
        plt.xlabel(x_axis_title)
    if y_axis_title!=None:
        plt.ylabel(y_axis_title)
    
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)
    plt.scatter(df_column_1, df_column_2)
  
    popt, pcov = curve_fit(linear_slope, df_column_1, df_column_2)
    username= "manott"
    server_name = "euler.ethz.ch"
    destination_path = r"/mnt/c/Users/leuna/OneDrive/Dokumente/8_Semester/Research_Project/Daten/Plots/"
    
    x = [-1, df_column_1.max()+1]
    
    df_column_1_ext = np.concatenate(([-1], df_column_1.values))
    df_column_1_ext = np.concatenate((df_column_1_ext, [10000]))
    y_pred = linear_slope(popt[0], df_column_1, popt[1])
    r_2 = r2_score(df_column_2, y_pred)
    mae = mean_absolute_error(df_column_2, y_pred)
    plt.plot(df_column_1_ext, linear_slope(popt[0], df_column_1_ext, popt[1]), 'r-',
         label='fit: m=%5.3f, a=%5.3f' % tuple(popt))
    plt.plot(x, x, "b:", label="parity line")
    plt.text(0.95, 0.95, f"R²: {r_2:.3f}", fontsize=12, ha="right", va="top",
             transform=plt.gca().transAxes)
    """
    scp_command = f"scp {username}@{server_name}:{plotname} {destination_path}"
    try:
        subprocess.run(scp_command, shell=True, check=True)
        print(f"✅ Plot successfully transferred to {destination_path}")
    except subprocess.CalledProcessError as e:
        print(f"SCP failed: {e}")"
    """

    plt.xlim(-1, df_column_1.max() +1)
    plt.ylim(-1, df_column_2.max() +1)

    plt.savefig(plotname)
    plt.clf()



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

