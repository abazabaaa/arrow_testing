from IPython import display
import os
import subprocess
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdShapeHelpers
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PyMol
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
import copy
from rich import print
import numpy as np
import pandas as pd
v = PyMol.MolViewer()
mols_ref = Chem.SDMolSupplier('/Users/tom/code_test_repository/arrow_testing/cdk2.sdf', removeHs=False)
cdk2mol_ref = [m for m in mols_ref]
cdk2mol_reference = cdk2mol_ref[0]
cdk2mol_reference
df = feather.read_feather('/Users/tom/code_test_repository/arrow_testing/test2.feather')
df = df.sort_values(by=['fps'], ascending=False)
df = df.reset_index(drop=True)
df[['name','conf','number']] = df.name.str.split('_', expand=True)
d2f = df.drop_duplicates(subset="name", keep="first")

# lis2 = df['fps']
d2f = d2f.reset_index(drop=True)
lis = d2f['mol_blocks']
d2f.head(100) 

mol_list = []
for m in lis:
    m1 = Chem.MolFromMolBlock(m, removeHs=False)
    mol_list.append(m1)
    
cdk2mol = [m for m in mol_list]
    
cdk2mol2 = copy.deepcopy(cdk2mol)
crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in cdk2mol2]
ref = cdk2mol_reference
ref_contrib = rdMolDescriptors._CalcCrippenContribs(ref)
targets = cdk2mol2[0:]
targets_contrib = crippen_contribs[0:]
 
for i, target in enumerate(targets):
    crippenO3A = rdMolAlign.GetCrippenO3A(target, ref, targets_contrib[i], ref_contrib)
    crippenO3A.Align()
    
v.DeleteAll()
v.ShowMol(ref, name='ref', showOnly=False)
for i in range(len(targets)):
    name = f'probe_{i}'
    v.ShowMol(targets[i], name=name, showOnly=False)
    
    
ref_mol_block = Chem.MolToMolBlock(ref_mol, removeHs=False)
hmols_1 = mol_list

crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in hmols_1]
crippen_ref_contrib = crippen_contribs[0]
crippen_prob_contribs = crippen_contribs[1:]
ref_mol1 = hmols_1[0]
prob_mols_1 = hmols_1[1:]


for i, target in enumerate(targets):
    crippenO3A = rdMolAlign.GetCrippenO3A(target, ref, targets_contrib[i], ref_contrib)
    crippenO3A.Align()
    
    
lig = Chem.SDMolSupplier('/Users/tom/d3_docking/lig.sdf', removeHs=False)
