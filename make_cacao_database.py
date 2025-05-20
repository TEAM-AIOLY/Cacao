import os
import pandas as pd
from scipy.io import savemat
import numpy as np

data_root ='C:/00_aioly/GitHub/Cacao/data/datasets/'    

nutrient_path = os.path.join(data_root,'Nutrients')
microbe_path = os.path.join(data_root,'microbial_and_ferti')

d3 =os.path.join(data_root,'3rd_sampling')
d1 =os.path.join(data_root,'1st_sampling')
d2 =os.path.join(data_root,'2nd_sampling')


database = {'nutrients': {}}
for fname in os.listdir(nutrient_path):
    if fname.endswith('.csv'):
        fpath = os.path.join(nutrient_path, fname)
        df = pd.read_csv(fpath, sep=';', header=0, index_col=0)
        key = os.path.splitext(fname)[0]  # Use file name without extension as key
        ref = df.index.astype(str).values
        value = df.iloc[:, 0].astype(float).values
        database['nutrients'][key] = {'ref': ref, 'value': value}
     

d_files = {
    'd1': {
        'spectral': 'spec_visnir_1st_sampling_rep.csv',
        'spad': 'SPAD_1st_sampling_rep.csv'
    },
    'd2': {
        'spectral': 'spec_visnir_2nd_sampling_rep.csv',
        'spad': 'SPAD_2nd_sampling_rep.csv'
    },
    'd3': {
        'spectral': 'spec_visnir_3rd_sampling_rep.csv',
        'spad': 'SPAD_3rd_sampling_rep.csv'
    }
}

d_paths = {'d1': d1, 'd2': d2, 'd3': d3}

for d_key in ['d1', 'd2', 'd3']:
#   if d_key=='d1':  
    database[d_key] = {}
    for data_type in ['spectral', 'spad']:
        fname = d_files[d_key][data_type]
        fpath = os.path.join(d_paths[d_key], fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep=';',  index_col=0)
            if data_type == 'spectral':
                # First row after index is wv, rest is value
                wv = df.columns.astype(float)
                ref = df.index.astype(str).values
                value = df.astype(float).values
                database[d_key][data_type] = {'ref': ref, 'wv': wv, 'value': value}
            else:  # spad
                ref = df.index.astype(str).values
                value = df.iloc[:, 0].astype(float).values
                database[d_key][data_type] = {'ref': ref, 'value': value}
        else:
            database[d_key][data_type] = None  # or handle missing file as needed
       
headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length',
           'Fresh_weight_leaves','Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight',
           'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']
           
for i, fname in enumerate(os.listdir(microbe_path)):
    if i == 0 and fname.endswith('.csv'):
        fpath = os.path.join(microbe_path, fname)
        df = pd.read_csv(fpath, sep=';', header=0,index_col=0)
        ref = df.index.astype(str).values
        headers = df.columns.tolist()
        value = df.values.astype(float)
        database['bio_rep'] = {
            'ref': ref,
            'headers': headers,
            'value': value
        }
          
    if i == 1 and fname.endswith('.csv'): 
        fpath = os.path.join(microbe_path, fname)
        df = pd.read_csv(fpath, sep=',', header=0)
        ref = df.iloc[:, 0].astype(str).values
        ferti = df.iloc[:, 1].astype(float).values
        bacteria = df.iloc[:, 2:6].astype(int).values
        database['treatments'] = {
            'ref': ref,
            'ferti': ferti,
            'bacteria': bacteria
        }
        
    if i == 3 and fname.endswith('.csv'):
        fpath = os.path.join(microbe_path, fname)
        df = pd.read_csv(fpath, sep=';', header=0)
        ref = df.iloc[:, 0].astype(str).values
        value = df.iloc[:, 1:].astype(int).values
        headers_growth = df.columns[1:].tolist()
        database['growth class'] = {
            'ref': ref,
            'headers': headers_growth,
            'value': value
        }


def replace_none_with_empty(obj):
    if isinstance(obj, dict):
        return {k: replace_none_with_empty(v) for k, v in obj.items() if v is not None}
    elif obj is None:
        return np.array([])
    else:
        return obj

database_clean = replace_none_with_empty(database)

save_path =os.path.join(data_root,'cacao_database.mat')      
savemat(save_path, database_clean)  