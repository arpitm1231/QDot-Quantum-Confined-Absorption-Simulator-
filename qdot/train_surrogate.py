import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('dataset.csv')
shapes = ['circle','square','hexagon','triangle']

# one-hot encode shape
for s in shapes:
    df[f'shape_{s}'] = (df['shape'] == s).astype(float)

feature_cols = [f'shape_{s}' for s in shapes] + ['size_nm','V0_eV','Eg_bulk','me','mh']
target_cols = ['Ee0','Eh0']

X = df[feature_cols].values
y = df[target_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

Xtr = x_scaler.transform(X_train)
Xte = x_scaler.transform(X_test)
ytr = y_scaler.transform(y_train)
yte = y_scaler.transform(y_test)

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    max_iter=6000,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=50,
)
model.fit(Xtr, ytr)

pred_scaled = model.predict(Xte)
pred = y_scaler.inverse_transform(pred_scaled)

mae_e = mean_absolute_error(y_test[:,0], pred[:,0])
mae_h = mean_absolute_error(y_test[:,1], pred[:,1])
r2_e = r2_score(y_test[:,0], pred[:,0])
r2_h = r2_score(y_test[:,1], pred[:,1])
print(f'Electron E0: MAE={mae_e*1000:.2f} meV, R2={r2_e:.4f}')
print(f'Hole     E0: MAE={mae_h*1000:.2f} meV, R2={r2_h:.4f}')

# derive wavelength error on test set for an interpretable metric
HC = 1239.84
Eg_bulk_test = X_test[:, feature_cols.index('Eg_bulk')]
gap_true = Eg_bulk_test + y_test[:,0] + y_test[:,1]
gap_pred = Eg_bulk_test + pred[:,0] + pred[:,1]
lam_true = HC / gap_true
lam_pred = HC / gap_pred
lam_mae = mean_absolute_error(lam_true, lam_pred)
print(f'Absorption wavelength MAE on held-out test set: {lam_mae:.1f} nm (range ~350-1200nm)')

# ---- export weights for vanilla-JS forward pass ----
def layer_list(arr_list):
    return [a.tolist() for a in arr_list]

export = {
    'feature_cols': feature_cols,
    'target_cols': target_cols,
    'shapes': shapes,
    'x_mean': x_scaler.mean_.tolist(),
    'x_scale': x_scaler.scale_.tolist(),
    'y_mean': y_scaler.mean_.tolist(),
    'y_scale': y_scaler.scale_.tolist(),
    'weights': layer_list(model.coefs_),
    'biases': layer_list(model.intercepts_),
    'activation': 'relu',
    'metrics': {
        'mae_Ee0_meV': round(mae_e*1000, 2),
        'mae_Eh0_meV': round(mae_h*1000, 2),
        'r2_Ee0': round(r2_e, 4),
        'r2_Eh0': round(r2_h, 4),
        'wavelength_mae_nm': round(lam_mae, 1),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
}
with open('surrogate_weights.json', 'w') as f:
    json.dump(export, f)

print('\nExported surrogate_weights.json')
print('Param count:', sum(a.size for a in model.coefs_) + sum(a.size for a in model.intercepts_))
