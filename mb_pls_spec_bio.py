import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS

import torch
import torch.nn.functional as F
from net.chemtools.metrics import ccc, r2_score
from scipy.io import loadmat


def snv(X):
    """Apply Standard Normal Variate (SNV) to each sample (row) of X."""
    X = np.asarray(X)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

def matstruct_to_dict(obj):
    if isinstance(obj, np.ndarray):
        return [matstruct_to_dict(o) for o in obj]
    elif hasattr(obj, '_fieldnames'):
        return {field: matstruct_to_dict(getattr(obj, field)) for field in obj._fieldnames}
    else:
        return obj

data_path = 'C:/00_aioly/GitHub/Cacao/data/datasets/cacao_database.mat'

database= loadmat(data_path, squeeze_me=True, struct_as_record=False)

database_dict = {k: v for k, v in database.items() if not k.startswith('__')}


mb_spec = []

for d_key in ['d1', 'd2', 'd3']:
    dset = matstruct_to_dict(database_dict[d_key])
    spectral = np.array(dset['spectral']['value'])
    mb_spec.append(spectral)

bio = matstruct_to_dict(database_dict['bio'])

X=mb_spec[2]
