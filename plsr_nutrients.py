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

nutirents =  dset = matstruct_to_dict(database_dict['nutrients'])

X=mb_spec[2]


w = 21  # Sav.Gol window size
p = 2   # Sav.Gol polynomial degree
d = 1   # Sav.Gol derivative order

# X = np.log1p(X)
# X = savgol_filter(X, window_length=w, polyorder=p, deriv=d, axis=1)
# X = detrend(X, axis=1)
# X=snv(X)

lv_list = [5, 5, 8, 3,8]

for i,key in enumerate(nutirents):
    perf=[]
    nutrient_data = nutirents[key]
    Y = np.asarray(nutrient_data['value'], dtype=np.float64).reshape(-1, 1)
    Y_rep = np.repeat(Y, 4, axis=0)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    # Example: every 3rd sample is test, others are train
    test_mask = (indices + 1) % 3 == 0
    X_train, X_test = X[~test_mask], X[test_mask]
    Y_train, Y_test = Y_rep[~test_mask], Y_rep[test_mask]
    
    
    n_components = 20
    plsr = PLS(ncomp=n_components)
    plsr.fit(X_train, Y_train)
    for lv in range(n_components):
        y_pred = plsr.predict(X_test, lv)
        rmse = torch.sqrt(F.mse_loss(y_pred, torch.from_numpy(Y_test), reduction='none')).mean(dim=0)
        perf.append(rmse)

    y_pred_final = plsr.predict(X_test, lv_list[i])
    ccc_value = ccc(torch.from_numpy(Y_test), y_pred_final)
    r2_value = r2_score(torch.from_numpy(Y_test), y_pred_final)

    # RMSEP curve
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, n_components + 1), [p.item() for p in perf], marker='o')
    plt.xlabel('Number of Latent Variables')
    plt.ylabel('RMSEP (Test)')
    plt.title(f'RMSEP Curve ({key})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Predicted vs True for test samples
    y_pred_final_np = y_pred_final.detach().cpu().numpy() if hasattr(y_pred_final, 'detach') else np.array(y_pred_final)
    y_test_np = Y_test
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test_np, y_pred_final_np, c='blue', label='Test samples')
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', label='1:1 line')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'Predicted vs True ({key})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()