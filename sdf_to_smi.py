
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter
print(rdBase.rdkitVersion)

 
mols = [mol for mol in Chem.SDMolSupplier('cdk2.sdf') if mol != None]
# make writer object with a file name.
writer = SmilesWriter('cdk2smi1.smi')
#Check prop names.
print(list(mols[0].GetPropNames()))

 
#SetProps method can set properties that will be written to files with SMILES.
writer.SetProps(['Cluster'])
#The way of writing molecules can perform common way.
for mol in mols:
    writer.write( mol )
writer.close()
