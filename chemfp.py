import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
import ray


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


@ray.remote
@stopwatch
def count_ligs(table_batch):
#     table_batch = table.read_next_batch()
#     df = table_batch.to_pandas()
#     print(f' the initial table_batch is type:{type(table_batch)}')
    smiles = list(table_batch.column('Smile'))
#     print(type(smiles))
#     print(smiles)
    count_ligs = len(smiles)
    # future = calc_similarity.remote(smiles, ref_smiles)
    
    mols = [Chem.MolFromSmiles(str(x)) for x in smiles]

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_ECFP4_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol,2)
    bulk_ECFP4_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in mols]
    similarity_efcp4 = [DataStructs.FingerprintSimilarity(ref_ECFP4_fps,x) for x in bulk_ECFP4_fps]
    
#     print(f' similarity_efcp4 is type {type(similarity_efcp4)}')
#     print(f' the table_batch is after sim search is type:{type(table_batch)}')
#     print(f'the number of columns in the table_batch: {table_batch.num_columns}')
#     print(f'the schema in the table_batch: {table_batch.schema}')
    sim_array = pa.array(similarity_efcp4)
#     print(f' similarity_efcp4 was coverted to type {type(sim_array)}')
    data = [
        table_batch[0],
        table_batch[1],
        sim_array
    ]
    batch = pa.RecordBatch.from_arrays(data, ['smiles', 'idnumber', 'sim_score'])
    df = batch.to_pandas()

    df1 = df[df['sim_score'] > 0.30]
    fields = [
    pa.field('smiles', pa.string()),
    pa.field('idnumber', pa.string()),
    pa.field('sim_score', pa.float64()),
    ]

    my_schema = pa.schema(fields)


    filtered_batch = pa.RecordBatch.from_pandas(df1, schema=my_schema, preserve_index=False)
    print(f' numbers of rows per record batch = {filtered_batch.num_rows}')
#     print(f'the final batch is of type: {type(batch)}')
    return filtered_batch
                        

                    
#     print(count_ligs)
#     print(i)
#     return df[df['Tanimoto_Similarity (ECFP4)'] > 0.7]


@stopwatch
def csv_chunk_extractor(chunks, chunksize, ref_smiles, include_columns):
    filename = '/Users/tom/code_test_repository/E_R_11_012_CNS4-6_2.smi'
    opts = pa.csv.ReadOptions(use_threads=True, block_size=chunksize)
    
    parse_options= pa.csv.ParseOptions(delimiter='\t')
    convert_options=pa.csv.ConvertOptions(include_columns=include_columns)
    table = pa.csv.open_csv(filename, opts, parse_options, convert_options)
    df_list = []
    ligs_counted = []
    #run with ray
    futures = [count_ligs.remote(chunk) for chunk in table]
    results = [ray.get(f) for f in futures]
    print(f'here is the type of a slice of the results: {type(results[0])}')
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

pa.set_cpu_count(2)
ray.init(num_cpus=7)

chunksize = 1048576*10
chunks = 10
ref_smiles = 'CCC1=CC(Cl)=C(OC)C(C(NC[C@H]2C[C@H](OC)CN2CC)=O)=C1O'
include_columns = ['Smile', 'CatalogID_1']
table_list = csv_chunk_extractor(chunks, chunksize, ref_smiles, include_columns)
ray.shutdown()
table = pa.Table.from_batches(table_list)
print(table.shape)
feather.write_feather(table, 'test2.feather')