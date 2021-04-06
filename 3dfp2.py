import copy
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools
from rdkit.Chem import AddHs, RemoveHs, MolFromSmiles, MolToPDBBlock
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
import ray
from rdkit.Chem import rdMolAlign
from e3fp.fingerprint.generate import fp, fprints_dict_from_mol
from e3fp.conformer.generate import generate_conformers
from e3fp.pipeline import fprints_from_mol

def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed


# @ray.remote
# def calc_similarity(smiles, ref_smiles):
#     mols = [Chem.MolFromSmiles(str(x)) for x in smiles]
#     ref_mol = Chem.MolFromSmiles(ref_smiles)
#     ref_ECFP4_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol,2)
#     bulk_ECFP4_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in mols]
#     similarity_efcp4 = [DataStructs.FingerprintSimilarity(ref_ECFP4_fps,x) for x in bulk_ECFP4_fps]
#     return similarity_efcp4

def gen_mol_blocks_from_confs(mols, num_confs, ref, ref_mol_block):

    
    # fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
    # ref_fprint = fprints_from_mol(ref_mol, fprint_params=fprint_params)
    # ref = ref_fprint[0].fold().to_rdkit()
    mols_b = copy.deepcopy(mols)
    names = []
    mol_blocks = []
    fps = []
    ref_mol_block = []
    for mol in mols_b:
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, ignoreSmoothingFailures=True, pruneRmsThresh=-1.0, maxAttempts=10 * num_confs, randomSeed=0xf00d)
        fprint_params = {'bits': 4096, 'first':num_confs, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
        fprints = fprints_from_mol(mol, fprint_params=fprint_params)
        binfp = [ fp.fold().to_rdkit() for fp in fprints ]
        similarity_efcp4 = [DataStructs.FingerprintSimilarity(ref,x) for x in binfp]
        for i in range(num_confs):
            sub_name = mol.GetProp('_Name')
            name = f'{sub_name}_confnum_{i}'
            names.append(name)
            mol_blocks.append(Chem.MolToMolBlock(mol, confId=i))
            fps.append(similarity_efcp4[i])
            ref_mol_block.append(ref_mol_block)
    df = pd.DataFrame(list(zip(names, mol_blocks, fps, ref_mol_block)),
               columns =['name', 'mol_blocks', 'fps', 'ref_mol_block'])
    return df

def align_mcs(mols, num_confs):
    suppl = [m for m in AllChem.SDMolSupplier('/Users/tom/code_test_repository/arrow_testing/cdk2.sdf', removeHs=False)]
    ref_mol = suppl[0]
    print(f'ref mol has atoms = {ref_mol.GetNumAtoms()}')
    mols_b = copy.deepcopy(mols)
    mol_blocks = []
    for mol in mols_b:
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
        mcs = rdFMCS.FindMCS([mol, ref_mol])
        smarts = mcs.smartsString
        match = Chem.MolFromSmarts(smarts)
        test_match_atoms = mol.GetSubstructMatch(match)
        ref_match_atoms = ref_mol.GetSubstructMatch(match)
        
        #Find alignments of all conformers of new drug to old drug:
        alignments_scores =[rdMolAlign.AlignMol(mol,
                        ref_mol,
                        prbCid=i,
                        atomMap=[[i,j] for i,j in zip(test_match_atoms, ref_match_atoms)]) for i in range(num_confs)]
        
        confId=int(np.argmin(alignments_scores))
        AllChem.CanonicalizeConformer(mol.GetConformer(confId))
        # print(Chem.MolToMolBlock(mol))
        mol_blocks.append(Chem.MolToMolBlock(mol))
    return pa.array(mol_blocks)
    # return alignments_scores


@ray.remote
@stopwatch
def count_ligs(table_batch, arr, ref_mol_block, num_confs=100):
#     table_batch = table.read_next_batch()
#     df = table_batch.to_pandas()
#     print(f' the initial table_batch is type:{type(table_batch)}')
    smiles = list(table_batch.column('SMILES'))
    names = list(table_batch.column('Name'))
    bitstring="".join(arr.astype(str))
    fp2 = DataStructs.cDataStructs.CreateFromBitString(bitstring)
    # print(list(fp2.GetOnBits()))
#     print(type(smiles))
    print(f'the number of smiles in the record batch is {len(smiles)}')
    count_ligs = len(smiles)
    # future = calc_similarity.remote(smiles, ref_smiles)
    
    mols = [Chem.MolFromSmiles(str(x)) for x in smiles]
    named_mols = []
    for count,m in enumerate(mols):
        molid = names[count]
        m.SetProp('_Name', str(molid))
        named_mols.append(m)

    # conf_mol_blocks = gen_conformers(mols, num_confs)
    # return conf_mol_blocks
    df = gen_mol_blocks_from_confs(named_mols, num_confs, fp2, ref_mol_block)
    
    # print(names)
    # print(mol_blocks)
    # print(fps)
    # data = [
    #     names,
    #     mol_blocks,
    #     fps
    # ]
    # batch = pa.RecordBatch.from_arrays(data, ['name_confidnum' 'mol_block', 'sim_score'])
    batch = pa.RecordBatch.from_pandas(df)
    return batch
    # MMFFOptimizeMoleculeConfs(mol)
    # mol = RemoveHs(mol)
    # mol_blocks = []
    # for i in range(num_confs):
    #     print(i)
    #     mol_blocks.append(Chem.MolToMolBlock(mol, confId=i))
    
    # ref_mol = Chem.MolFromSmiles(ref_smiles)
    # ref_ECFP4_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol,2)

    
    # m = Chem.MolFromMolBlock(mol_blocks[0])
    # print(f' the mol is not a mol true or false: {m is None}')
    # return mols
#     print(f' similarity_efcp4 is type {type(similarity_efcp4)}')
#     print(f' the table_batch is after sim search is type:{type(table_batch)}')
#     print(f'the number of columns in the table_batch: {table_batch.num_columns}')
#     print(f'the schema in the table_batch: {table_batch.schema}')
    # mol_block_array = pa.array(mol_blocks)
#     print(f' similarity_efcp4 was coverted to type {type(sim_array)}')

    # df = batch.to_pandas()

    # df1 = df[df['sim_score'] > 0.30]
    # fields = [
    # pa.field('smiles', pa.string()),
    # pa.field('idnumber', pa.string()),
    # pa.field('sim_score', pa.float64()),
    # ]

    # my_schema = pa.schema(fields)


    # filtered_batch = pa.RecordBatch.from_pandas(df1, schema=my_schema, preserve_index=False)
    # print(f' numbers of rows per record batch = {filtered_batch.num_rows}')
#     print(f'the final batch is of type: {type(batch)}')
    # return mol_block_array
                        

# @stopwatch
# def align_mcs(new_mol, ref_mol, confs):
#     """Ref mol is the crystallized ligand. New mol is the drug you want to add to the structure.
#     It returns an array of scores (one for each conformer), where the lowest score is best"""
#     ##Find maximum common substructure so we can align based on this:
#     mcs = Chem.rdFMCS.FindMCS([new_mol, ref_mol])
#     smarts = mcs.smartsString
#     match = Chem.MolFromSmarts(smarts)
#     test_match_atoms = new_mol.GetSubstructMatch(match)
#     ref_match_atoms = ref_mol.GetSubstructMatch(match)
    
#     #Find alignments of all conformers of new drug to old drug:
#     alignments_scores =[rdMolAlign.AlignMol(new_mol,
#                     ref_mol,
#                     prbCid=i,
#                     atomMap=[[i,j] for i,j in zip(test_match_atoms, ref_match_atoms)]) for i in range(confs)]
    
#     return alignments_scores

# @stopwatch
# def get_lowest_energy_conf_from_align_MCS_with_xtal_ligand(smiles, xtal_sdf_path):
#   r = Chem.MolFromSmiles(smiles)
#   e = Chem.MolFromMolFile(xtal_sdf_path, removeHs=False)
#   r2=Chem.AddHs(r)
#   # r3= copy.deepcopy(r2)
#   # AllChem.EmbedMolecule(r3,useRandomCoords=True)
#   generate_conformers(r2, n=1000)
#   alignments = align_mcs(r2, e, 1000)
#   confId=int(np.argmin(alignments))
#   # mb = Chem.MolToMolBlock(r2, confId=confId)
#   # y = Chem.MolFromMolBlock(mb)
#   AllChem.CanonicalizeConformer(r2.GetConformer(confId))
#   return r2, e
                    
#     print(count_ligs)
#     print(i)
#     return df[df['Tanimoto_Similarity (ECFP4)'] > 0.7]


@stopwatch
def csv_chunk_extractor(chunks, chunksize, include_columns, arr, ref_mol_block):
    filename = '/Users/tom/code_test_repository/arrow_testing/cdk2smi1.smi'
    opts = pa.csv.ReadOptions(use_threads=True, block_size=chunksize)
    
    parse_options= pa.csv.ParseOptions(delimiter=' ')
    convert_options=pa.csv.ConvertOptions(include_columns=include_columns)
    table = pa.csv.open_csv(filename, opts, parse_options, convert_options)
    df_list = []
    ligs_counted = []
    #run with ray
    futures = [count_ligs.remote(chunk, arr, ref_mol_block) for chunk in table]
    results = [ray.get(f) for f in futures]
    print(f'here is the type of a slice of the results: {type(results)}')
    # run single threaded
#     results = [count_ligs(chunk) for chunk in table]
#     print(results)
    return results
#     for i in range(chunks):
#         table_batch = table.read_next_batch()
#         futures = count_ligs.remote(table_batch)
#         df_list.append(df2)
#         ligs_counted.append(count_ligs)
#     Sum = sum(ligs_counted)
#     print(Sum)
#     return df_list

# @ray.remote
# class CountedSlows:
#     def __init__(self, initial_count = 0):
#         self.count = initial_count
#     def slow(self, record):
#         self.count += 1
#         new_record = expensive_process(record)
#         return new_record
#     def get_count(self):
#         return self.count
    
# cs = CountedSlows.remote() # Note how actor construction works
# futures = [cs.slow.remote(r) for r in records]

# while len(futures) > 0:
#      finished, rest = ray.wait(futures)
#      value = ray.get(finished[0])
#      print(value)
#      futures = rest

# count_future_id = cs.get_count.remote()
# ray.get(count_future_id)

pa.set_cpu_count(1)
ray.init(num_cpus=4)

# mols = [m for m in Chem.SDMolSupplier('cdk2.sdf')]
# for m in mols:
#     molid = m.GetProp('id')
#     m.SetProp('_Name', molid) #_Name prop is required for align with shape-it
# ref = Chem.Mol(mols[0])

chunksize = 1048576/10000
chunks = 10

suppl = [m for m in AllChem.SDMolSupplier('/Users/tom/code_test_repository/arrow_testing/cdk2.sdf', removeHs=False)]
ref_mol = suppl[0]
ref_mol_block = Chem.MolToMolBlock(ref_mol)
fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
fprints = fprints_from_mol(ref_mol, fprint_params=fprint_params)
binfp = [ fp.fold().to_rdkit() for fp in fprints ]
arr = np.zeros((0,), dtype=np.int8)

DataStructs.ConvertToNumpyArray(binfp[0],arr)

# ref_fprint = fprints_from_mol(ref_mol, fprint_params=fprint_params)
# ref = ref_fprint[0].fold().to_rdkit()
# ref_smiles = 'CCC1=CC(Cl)=C(OC)C(C(NC[C@H]2C[C@H](OC)CN2CC)=O)=C1O'
include_columns = ['SMILES', 'Name']
table_list = csv_chunk_extractor(chunks, chunksize, include_columns, arr, ref_mol_block)
ray.shutdown()
print('finished alignment')
# table = Chem.MolFromMolBlock(mol_block_array[0])
# print(f' the mol from the array is not a mol true or false: {m_out_of_array is None}')

table = pa.Table.from_batches(table_list)
print(table.shape)
feather.write_feather(table, 'test2.feather')